#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import numpy as np
import ox.common.IR.graph_pb2 as graph_pb2
from ox.common.IR.graph_pb2 import ModelDef, NodeDef, GraphDef, DataType

info_model = {
    'doc_url': '*',
    'contributor_name': '*',
    'contributor_email': '*',
    'contributor_institute': '*',
    'framework_name': '*',
    'framework_version': '*',
    'model_name': '*',
    'model_version': '*',
    'version': '*'
}

class Parser(object):

    def __init__(self):
        self.IR_model = ModelDef()
        self._add_model_info()
        self.IR_graph = self.IR_model.graph
        self.weight_loaded = False

        # name --> (weight_name --> ndarray)
        self.weights = dict()


    def run(self, dest_path):
        op_sets = self.gen_IR()
        self.save_to_json(dest_path + ".json")
        self.save_to_proto(dest_path + ".pb")
        self.save_weights(dest_path + ".npy")

        return op_sets
    
    def _add_model_info(self, model=info_model):
        self.IR_model.doc_url = model['doc_url']
        self.IR_model.framework_name = model['framework_name']
        self.IR_model.framework_version = model['framework_version']
        self.IR_model.model_name = model['model_name']
        self.IR_model.model_version = model['model_version']
        self.IR_model.version = model['version']
        # Contributor
        self.IR_model.contributors.name.append(model['contributor_name'])
        self.IR_model.contributors.email.append(model['contributor_email'])
        self.IR_model.contributors.institute.append(model['contributor_institute'])


    @property
    def src_graph(self):
        raise NotImplementedError


    def get_son(self, name, path, set_flag = False):
        return self.src_graph.get_son(name, path, set_flag)


    def get_parent(self, name, path, set_flag = False):
        return self.src_graph.get_parent(name, path, set_flag)


    def set_weight(self, layer_name, weight_name, data):
        if not layer_name in self.weights:
            self.weights[layer_name] = dict()
        layer = self.weights[layer_name]
        layer[weight_name] = data


    def save_to_json(self, filename):
        import google.protobuf.json_format as json_format
        json_str = json_format.MessageToJson(self.IR_model, preserving_proto_field_name = True)

        with open(filename, "w") as of:
            of.write(json_str)

        print ("IR network structure is saved as [{}].".format(filename))

        return json_str


    def save_to_proto(self, filename):
        proto_str = self.IR_model.SerializeToString()
        with open(filename, 'wb') as of:
            of.write(proto_str)

        print ("IR network structure is saved as [{}].".format(filename))

        return proto_str


    def save_weights(self, filename):
        if self.weight_loaded:
            with open(filename, 'wb') as of:
                np.save(of, self.weights)
            print ("IR weights are saved as [{}].".format(filename))

        else:
            print ("Warning: weights are not loaded.")


    def convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: end_idx = len(source_node.in_edges)
        for idx in range(start_idx, end_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name.lstrip('_'))


    @staticmethod
    def channel_first_conv_kernel_to_IR(tensor):
        dim = tensor.ndim
        tensor = np.transpose(tensor, list(range(2, dim)) + [1, 0])
        return tensor


    @staticmethod
    def channel_first_shape_to_IR(shape):
        return [shape[0]] + list(shape[2:]) + [shape[1]]

    @staticmethod
    def channel_first_axis_to_IR(index):
        if index == 0:
            return 0
        elif index == 1:
            return -1
        else:
            return index - 1
