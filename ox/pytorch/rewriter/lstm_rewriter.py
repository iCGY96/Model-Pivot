import numpy as np
import re
from ox.pytorch.pytorch_graph import DYNAMIC_SHAPE_MARK
from ox.pytorch.pytorch_graph import PytorchGraphNode


class LstmRewriter(object):
    """
    Re-write Pytorch onnx::lstm op into the equivalent detail ops, that is,
    MatMul, add, Sigmoid, Tanh, etc.

    The detail document of onnx::lstm is at
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM

    Note: we only handle singe-directional lstm here and use defualt peephole
    weight (0). The support for those will be TODO.
    """
    def __init__(self, pytorch_graph):
        self.pytorch_graph = pytorch_graph
        self.unique_id = 0

    def get_next_unique_id(self):
        self.unique_id += 1
        return self.unique_id

    def run(self):
        self.rewrite_graph()
        return self.pytorch_graph

    def rewrite_graph(self):
        replace_nodes = []
        for layer in self.pytorch_graph.topological_sort:
            current_node = self.pytorch_graph.get_node(layer)
            onnx_node_type = current_node.type
            if onnx_node_type == "onnx::LSTM":
                replace_nodes.append(current_node)

        for lstm_id, lstm_node in enumerate(replace_nodes):
            self.process_lstm(lstm_node, lstm_id)
            self.pytorch_graph.remove_node(lstm_node)

        self.remove_useless_node()

    def remove_useless_node(self):
        remove_node = []
        for layer in self.pytorch_graph.topological_sort:
            current_node = self.pytorch_graph.get_node(layer)
            if len(current_node.in_edges) == 0 and len(
                    current_node.out_edges) == 0:
                remove_node.append(current_node)
        for node in remove_node:
            self.pytorch_graph.remove_node(node)

    def process_lstm(self, lstm_node, lstm_id):
        """
        Inserts detail nodes that are equivalent to lstm node.
        """
        lstm_inputs = lstm_node.in_edges

        x_name = lstm_inputs[0]
        x_node = self.pytorch_graph.get_node(x_name)
        x_shape = self.pytorch_graph.shape_dict[x_name]
        # x_shape is [seq_len, batch_size, input_size]
        seq_len = x_shape[0]
        batch_size = x_shape[1]

        w_iofg = self.insert_weight_node(lstm_node, lstm_inputs[1], lstm_id, is_param_w=True)
        r_iofg = self.insert_weight_node(lstm_node, lstm_inputs[2], lstm_id, is_param_w=False)
        wb_iofg, rb_iofg = self.insert_bias_node(lstm_node, lstm_id)
        h_last, c_last = self.insert_initial_state_node(lstm_node, batch_size)

        outputs = []
        for i in range(seq_len):
            # input at i-th step, that is x[i]
            x_t_node = self.insert_x_t_node(lstm_node, i)

            # input gates
            i_t_node = self.insert_iofg_node(lstm_node, x_t_node, w_iofg[0],
                                             h_last, r_iofg[0], wb_iofg[0],
                                             rb_iofg[0], "Sigmoid")
            # output gates
            o_t_node = self.insert_iofg_node(lstm_node, x_t_node, w_iofg[3],
                                             h_last, r_iofg[3], wb_iofg[3],
                                             rb_iofg[3], "Sigmoid")
            # forget gates
            f_t_node = self.insert_iofg_node(lstm_node, x_t_node, w_iofg[1],
                                             h_last, r_iofg[1], wb_iofg[1],
                                             rb_iofg[1], "Sigmoid")
            # cell gates
            g_t_node = self.insert_iofg_node(lstm_node, x_t_node, w_iofg[2],
                                             h_last, r_iofg[2], wb_iofg[2],
                                             rb_iofg[2], "Tanh")

            # update states
            c_t_node = self.insert_update_c_node(lstm_node, c_last, f_t_node,
                                                 i_t_node, g_t_node)
            h_t_node = self.insert_update_h_node(lstm_node, o_t_node, c_t_node)
            h_last = h_t_node
            c_last = c_t_node

            outputs.append(h_t_node)

        hidden_size = lstm_node.attrs["hidden_size"]
        concat_attrs = {"axis": 0}
        output_node = self.insert_node(lstm_node.name, "onnx::Concat",
                                       [seq_len, batch_size, hidden_size],
                                       concat_attrs)
        for o in outputs:
            self.pytorch_graph._make_connection(o.name, output_node.name)

        return output_node, h_last, c_last

    def insert_x_t_node(self, lstm_node, i):
        """
        Insert x_t_node representing x[i] where x is the lstm_node.in_edges[0]
        """
        lstm_inputs = lstm_node.in_edges

        x_name = lstm_inputs[0]
        x_shape = self.pytorch_graph.shape_dict[x_name]
        num_id = re.search(r"[\d]+", lstm_node.name).group(0)

        # Slice x from [seq_len, batch_size, input_size] to [1, batch_size, input_size]
        x_slice_output_shape = x_shape[:]
        x_slice_output_shape[0] = 1
        x_slice_attrs = {"axes": [0], "starts": [i], "ends": [i + 1]}
        x_slice_node = self.insert_node(lstm_node.name, "onnx::Slice",
                                        x_slice_output_shape, x_slice_attrs)
        self.pytorch_graph._make_connection(x_name, x_slice_node.name)

        # Squeeze the x_slice from [1, batch_size, input_size] to [batch_size, input_size]
        x_t_output_shape = x_slice_output_shape[1:]
        x_t_attrs = {"axes": [0]}
        x_t_node = self.insert_node(lstm_node.name, "onnx::Squeeze",
                                    x_t_output_shape, x_t_attrs)
        self.pytorch_graph._make_connection(x_slice_node.name, x_t_node.name)
        return x_t_node

    def insert_weight_node(self, lstm_node, w_name, lstm_id, is_param_w):
        """
        Slice the input weight with w_name into w[iofg] where weight can be
        weight or recurrence_weight of lstm.
        """
        hidden_size = lstm_node.attrs["hidden_size"]
        output_shape = self.pytorch_graph.shape_dict[w_name][:]  # make a copy

        # Slice the weight size from [4 * hidden_size, input_size] to [hidden_size, input_size]
        if output_shape != DYNAMIC_SHAPE_MARK:
            output_shape[0] /= 4

        w_iofg = []
        weight_param_name = "lstm.weight_"
        if is_param_w:
            weight_param_name += "ih_"
        else:
            weight_param_name += "hh_"
        weight_param_name += ("l" + str(lstm_id))
        for i in range(4):
            w_attrs = {
                "axes": [0],
                "starts": [i * hidden_size],
                "ends": [(i + 1) * hidden_size],
                "input_from_param": weight_param_name
            }
            w_node = self.insert_node(lstm_node.name, "onnx::Slice",
                                      output_shape, w_attrs)
            self.pytorch_graph._make_connection(w_name, w_node.name)
            w_iofg.append(w_node)

        return w_iofg

    def insert_bias_node(self, lstm_node, lstm_id):
        lstm_inputs = lstm_node.in_edges
        hidden_size = lstm_node.attrs["hidden_size"]

        bias_name = lstm_inputs[3]
        output_shape = self.pytorch_graph.shape_dict[
            bias_name][:]  # make a copy

        # Slice the bias size from [8 * hidden_size] to [hidden_size]
        if output_shape != DYNAMIC_SHAPE_MARK:
            output_shape[0] /= 8

        sliced_bias = []
        for i in range(8):
            b_attrs = {
                "axes": [0],
                "starts": [i * hidden_size],
                "ends": [(i + 1) * hidden_size],
                "input_from_param": ["lstm.bias_ih_l" + str(lstm_id), "lstm.bias_hh_l" + str(lstm_id)]
            }
            b_node = self.insert_node(lstm_node.name, "onnx::Slice",
                                      output_shape, b_attrs)
            self.pytorch_graph._make_connection(bias_name, b_node.name)
            sliced_bias.append(b_node)

        wb_iofg = sliced_bias[0:4]
        rb_iofg = sliced_bias[4:8]
        return wb_iofg, rb_iofg

    def insert_initial_state_node(self, lstm_node, batch_size):
        lstm_inputs = lstm_node.in_edges
        hidden_size = lstm_node.attrs["hidden_size"]

        output_shape = [batch_size, hidden_size]
        default_attrs = {"value": 0.0, "full_shape": output_shape}
        if len(lstm_inputs) >= 6:
            initial_h = self.pytorch_graph.get_node(lstm_inputs[5])
        else:
            initial_h = self.insert_node(lstm_node.name, "onnx::Constant",
                                         output_shape, default_attrs)

        if len(lstm_inputs) >= 7:
            initial_c = self.pytorch_graph.get_node(lstm_inputs[6])
        else:
            initial_c = self.insert_node(lstm_node.name, "onnx::Constant",
                                         output_shape, default_attrs)
        return initial_h, initial_c

    def insert_update_h_node(self, lstm_node, o_t_node, c_t_node):
        """
        Updates h in lstm. The formula is "h_t = o_t (.) Tanh(c_t)".
        """
        c_shape = self.pytorch_graph.shape_dict[c_t_node.name][:]
        tanh_c_node = self.insert_node(lstm_node.name, "onnx::Tanh", c_shape)
        self.pytorch_graph._make_connection(c_t_node.name, tanh_c_node.name)

        h_t_node = self.insert_elementwise_op_node(lstm_node, o_t_node,
                                                   tanh_c_node, "Mul")
        return h_t_node

    def insert_update_c_node(self, lstm_node, last_c_node, f_t_node, i_t_node,
                             g_t_node):
        """
        Updates c in lstm. The fomula is "c_t = f_t (.) last_c + i_t (.) g_t"
        """
        mul_c_f_node = self.insert_elementwise_op_node(lstm_node, f_t_node,
                                                       last_c_node, "Mul")
        mul_i_g_node = self.insert_elementwise_op_node(lstm_node, i_t_node,
                                                       g_t_node, "Mul")

        c_t_node = self.insert_elementwise_op_node(lstm_node, mul_c_f_node,
                                                   mul_i_g_node, "Add")
        return c_t_node

    def insert_iofg_node(self, lstm_node, x_node, w_node, h_node, r_node,
                         wb_node, rb_node, act_func):
        """
        Inserts a node to compute iofg, that is input, or forget, or cell, or output gates of lstm.

        The fomula is "act_func( (x * w^T + wb) + (h * r^T + rb) )"
        """
        w_mul_add_bias = self.insert_mul_add_bias_node(lstm_node, x_node,
                                                       w_node, wb_node)
        r_mul_add_bias = self.insert_mul_add_bias_node(lstm_node, h_node,
                                                       r_node, rb_node)

        w_mul_add_bias_shape = self.pytorch_graph.shape_dict[
            w_mul_add_bias.name][:]
        r_mul_add_bias_shape = self.pytorch_graph.shape_dict[
            r_mul_add_bias.name][:]

        add_node = self.insert_elementwise_op_node(lstm_node, w_mul_add_bias,
                                                   r_mul_add_bias, "Add")
        add_shape = self.pytorch_graph.shape_dict[add_node.name][:]

        if act_func.lower() == "sigmoid":
            output_node = self.insert_node(lstm_node.name, "onnx::Sigmoid",
                                           add_shape)
            self.pytorch_graph._make_connection(add_node.name,
                                                output_node.name)
        elif act_func.lower() == "tanh":
            output_node = self.insert_node(lstm_node.name, "onnx::Tanh",
                                           add_shape)
            self.pytorch_graph._make_connection(add_node.name,
                                                output_node.name)
        else:
            raise ValueError("Input '" + str(act_func) +
                             "' is not supported activation function.")

        return output_node

    def insert_elementwise_op_node(self, lstm_node, left_node, right_node,
                                   op_str):
        left_shape = self.pytorch_graph.shape_dict[left_node.name][:]
        right_shape = self.pytorch_graph.shape_dict[right_node.name][:]
        mul_shape = DYNAMIC_SHAPE_MARK
        if left_shape != DYNAMIC_SHAPE_MARK:
            mul_shape = left_shape
        elif right_shape != DYNAMIC_SHAPE_MARK:
            mul_shape = right_shape

        if op_str not in ["Add", "Mul"]:
            raise ValueError("op_str '" + str(op_str) +
                             "' is not supported elementwise op.")
        mul_node = self.insert_node(lstm_node.name, "onnx::" + op_str,
                                    mul_shape)
        self.pytorch_graph._make_connection(left_node.name, mul_node.name)
        self.pytorch_graph._make_connection(right_node.name, mul_node.name)
        return mul_node

    def insert_mul_add_bias_node(self, lstm_node, x_node, w_node, b_node):
        """
        Inserts a node representing "x_node * (w_node ^ T) + b_node"
        """
        w_trans_shape = self.pytorch_graph.shape_dict[w_node.name][:]
        if w_trans_shape != DYNAMIC_SHAPE_MARK:
            w_trans_shape.reverse()
        perm_attrs = {"perm_list" : [0, 1]}
        w_trans_node = self.insert_node(lstm_node.name, "onnx::Transpose",
                                        w_trans_shape, perm_attrs)
        self.pytorch_graph._make_connection(w_node.name, w_trans_node.name)

        x_shape = self.pytorch_graph.shape_dict[x_node.name][:]
        if x_shape == DYNAMIC_SHAPE_MARK or w_trans_shape == DYNAMIC_SHAPE_MARK:
            mul_shape = DYNAMIC_SHAPE_MARK
        else:
            mul_shape = [x_shape[0], w_trans_shape[1]]
        mul_node = self.insert_node(lstm_node.name, "onnx::MatMul", mul_shape)
        self.pytorch_graph._make_connection(x_node.name, mul_node.name)
        self.pytorch_graph._make_connection(w_trans_node.name, mul_node.name)

        add_node = self.insert_elementwise_op_node(lstm_node, mul_node, b_node,
                                                   "Add")
        return add_node

    def insert_node(self,
                    lstm_scope_name,
                    onnx_type,
                    output_shape=DYNAMIC_SHAPE_MARK,
                    attrs={}):
        """
        Utility function to insert a node into lstm scope.
        """
        unique_id = self.get_next_unique_id()
        assert onnx_type.startswith(
            "onnx::"), "Input '%s' is not onnx_type" % (onnx_type)
        name_id = "__" + onnx_type[6:] + "_" + str(unique_id)
        name = lstm_scope_name + name_id
        node = PytorchGraphNode(name, onnx_type, name_id, attrs)
        self.pytorch_graph.shape_dict[node.name] = output_shape
        self.pytorch_graph.layer_map[node.name] = node
        self.pytorch_graph.layer_name_map[node.name] = node.name

        return node
