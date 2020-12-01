#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from ox.common.DataStructure.graph import GraphNode, Graph
import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict
import numpy as np
import re

DYNAMIC_SHAPE_MARK = "Dynamic"

class PytorchGraphNode(GraphNode):


    def __init__(self, *args):
        if len(args) != 1:
            self.consturct_from_fields(args)
            return

        layer = args[0]
        self._name = layer.scopeName()
        self._kind = layer.kind()
        import re
        node_id = re.search(r"[\d]+", layer.__str__())
        self.id = node_id.group(0)

        super(PytorchGraphNode, self).__init__(layer)

        # print(self._kind)

        if 'Constant' in self._kind:
            idx1 = layer.__str__().find('value')
            idx2 = layer.__str__().find('[')
            for i in range(idx1, len(layer.__str__())):
                if layer.__str__()[i] in '[}]': idx2 = i; break
            val = re.findall(r'\d+\.\d+|\d+|\-\d+\.\d+|\-\d+', layer.__str__()[idx1:idx2])
            if 'Long' in layer.__str__():
               for i in range(len(val)):
                    val[i] = int(val[i])
            else:
                for i in range(len(val)):
                    val[i] = float(val[i])
            if 'LongTensor' in layer.__str__():
                # val[0] = -1
                self.attrs = {k : val for k in layer.attributeNames()}
            else:
                self.attrs = {k : val for k in layer.attributeNames()}
        else:
            self.attrs = {k : layer[k] for k in layer.attributeNames()}

        self.weights_name = '.'.join(
            re.findall(r'\[([\w\d.]+)\]', self._name)
        )

    def consturct_from_fields(self, args):
        self._name = args[0]
        self._kind = args[1]
        self.id = args[2]
        super(PytorchGraphNode, self).__init__(layer=None)
        self.attrs = args[3] if len(args) >= 4 else {}
        self.weights_name = '.'.join(
            re.findall(r'\[([\w\d.]+)\]', self._name)
        )

    @property
    def name(self):
        name = self._name + self.id
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        name = name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
        return name

    @property
    def type(self):
        return self._kind

    @property
    def pytorch_layer(self):
        return self.layer

    def __str__(self):
        return "PytorchGraphNode name: %s, type: %s, attrs: %s" % (self.name, self.type, str(self.attrs))

class PytorchGraph(Graph):

    def __init__(self, model):
        # sanity check.
        super(PytorchGraph, self).__init__(model)
        self.model = model
        self.state_dict = _unique_state_dict(self.model)
        self.shape_dict = dict()


    @staticmethod
    def _optimize_graph(graph, aten, export_raw_ir=False):
        # run dce first to eliminate dead parts of the graph that might have been
        # left behind by things like symbolic_override

        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_peephole(graph)
        torch._C._jit_pass_lint(graph)
        if not export_raw_ir:
            graph = torch._C._jit_pass_onnx(graph, aten)
            torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_onnx_peephole(graph)
            torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        return graph


    @staticmethod
    def get_node_id(node):
        node_id = re.search(r"[\d]+", node.__str__())
        return node_id.group(0)

    @contextlib.contextmanager
    def set_training(self, model, mode):
        r"""
        A context manager to temporarily set the training mode of 'model'
        to 'mode', resetting it when we exit the with-block.  A no-op if
        mode is None.
        """
        if mode is None:
            yield
            return
        old_mode = model.training
        if old_mode != mode:
            model.train(mode)
        try:
            yield
        finally:
            if old_mode != mode:
                model.train(old_mode)


    def build(self, shape):
        """
        build graph for pytorch 0.4.0
        """

        import re
        # construct graph
        dummy_input = torch.autograd.Variable(torch.randn(shape), requires_grad=False)


        with self.set_training(self.model, False):
            trace, output = torch.jit.get_trace_graph(self.model, (dummy_input, ))

        trace.set_graph(PytorchGraph._optimize_graph(trace.graph(), False))

        # nodes
        nodes = list(trace.graph().nodes())


        # input layer
        # TODO



        # build each layer
        for node in nodes:

            node_id = PytorchGraph.get_node_id(node)
            node_scope = node.scopeName()
            node_name = node_scope + node_id
            node_name = node_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
            output_shape_str = re.findall(r'[^()!]+', node.__str__())[1]
            # print(node.__str__())
            if 'Constant' in node.__str__():
                # print(node.__str__())
                idx1 = node.__str__().find('value')
                idx2 = node.__str__().find('[')
                for i in range(idx1, len(node.__str__())):
                    if node.__str__()[i] in '[}]': idx2 = i; break
                val = re.findall(r'\d+\.\d+|\d+', node.__str__()[idx1:idx2])
                for i in range(len(val)):
                    val[i] = float(val[i])
                output_shape = list(np.array(val).shape)
            else:
                output_shape = []
                for x in output_shape_str.split(', '):
                    x_str = x.replace('!', '')
                    if x_str.isnumeric():
                        output_shape.append(int(x_str))
                    else:
                        # RNN or other control flow related node has
                        # Changable output shape so it is undefined
                        output_shape = DYNAMIC_SHAPE_MARK
                        break

            self.shape_dict[node_name] = output_shape
            self.layer_map[node_name] = PytorchGraphNode(node)
            self.layer_name_map[node_name] = node_name

            # input
            for node_input in list(node.inputs()):

                if PytorchGraph.get_node_id(node_input.node()) and node_input.node().scopeName():
                    node_input_name = node_input.node().scopeName() + PytorchGraph.get_node_id(node_input.node())
                    node_input_name = node_input_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
                    self._make_connection(node_input_name, node_name)
                    # print(node_input_name ,'->', node_name)


        super(PytorchGraph, self).build()

    def remove_node(self, node):
        del self.shape_dict[node.name]
        del self.layer_map[node.name]
        del self.layer_name_map[node.name]
        for in_node_name in node.in_edges:
            in_node = self.get_node(in_node_name)
            in_node.out_edges.remove(node.name)
        for out_node_name in node.out_edges:
            out_node = self.get_node(out_node_name)
            out_node.in_edges.remove(node.name)
        node.in_edges = []
        node.out_edges = []
        self.input_layers = []
        super(PytorchGraph, self).rebuild()

