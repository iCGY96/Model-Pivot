import os
import sys
sys.path.append('./../')
import common.IR.model_pb2 as model_pb2
from common.IR.model_pb2 import TensorShape, Attribute
from common.IR.graph import GraphClass, NodeClass


def load_protobuf_from_file(container, filename):
    with open(filename, 'rb') as fin:
        file_content = fin.read()

    # First try to read it as a binary file.
    try:
        container.ParseFromString(file_content)
        print("Parse file [%s] with binary format successfully." % (filename))
        return container

    except Exception as e:  # pylint: disable=broad-except
        print ("Info: Trying to parse file [%s] with binary format but failed with error [%s]." % (filename, str(e)))

    # Next try to read it as a text file.
    try:
        from google.protobuf import text_format
        text_format.Parse(file_content.decode('UTF-8'), container, allow_unknown_extension=True)
        print("Parse file [%s] with text format successfully." % (filename))
    except text_format.ParseError as e:
        raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

    return container


class IRGraphNode(NodeClass):

    @staticmethod
    def replace_scope(name):
        return name.replace('/', '_')

    @property
    def IR_layer(self):
        return self.layer

    @property
    def name(self):
        return self.layer.name

    @property
    def type(self):
        return self.layer.operator

    def set_attrs(self, attrs):
        assign_IRnode_values(self, attrs)


    def get_attr(self, name, default_value = None):
        if name in self.layer.attribute:
            attr = self.layer.attribute[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if isinstance(val, Attribute.ListValue):
                return list(val.ListFields()[0][1])
            else:
                return val.decode('utf-8') if isinstance(val, bytes) else val
        else:
            return default_value


class IRGraph(GraphClass):

    @staticmethod
    def shapeToStr(tensor_shape, keep_minus_one = False):
        ret = ""
        first = True
        for e in tensor_shape.dim:
            if e.size != -1 or keep_minus_one:
                if first == False:
                    ret += ", "
                ret += str(e.size)
                first = False
        return ret


    def __init__(self, filename):
        model = model_pb2.Model()
        load_protobuf_from_file(model, filename)
        super(IRGraph, self).__init__(model)


    def filter_node(self):
        self.layer_map = dict(filter(lambda layer: layer[1].in_edges or layer[1].out_edges, self.layer_map.items()))


    def build(self):
        for layer in self.model.graph.node:
            self.layer_map[layer.name] = IRGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name

        for i, layer in enumerate(self.model.graph.node):
            for pred in layer.input:
                self._make_connection(pred, layer.name)

        self.filter_node()
        super(IRGraph, self).build()