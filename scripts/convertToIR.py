import os
import sys
sys.path.append('./../')
import google.protobuf.text_format as text_format
from six import text_type as _text_type

info_model = {
    'doc_url': '*',
    'contributor_name': '*',
    'contributor_email': '*',
    'contributor_institute': '*',
    'framework_name': '*',
    'framework_version': '*',
    'model_name': 'test model',
    'model_version': '0.0.1',
    'version': '0.1.0'
}

def _convert(args):
    # check input shape
    if args.inputShape != None:
        inputshape = []
        for x in args.inputShape:
            shape = x.split(',')
            inputshape.append([int(x) for x in shape])
    else:
        inputshape = [None]
    
    info_model['framework_name'] = args.src
    if args.src == 'caffe':
        from common.caffe.transformer import CaffeTransformer
        transformer = CaffeTransformer(args.network, args.weights, "tensorflow", inputshape[0], phase = args.caffePhase)
        graph = transformer.transform_graph()
        data = transformer.transform_data()

        from common.caffe.writer import JsonFormatter, ModelSaver, PyWriter
        JsonFormatter(graph, info_model).dump(args.dstPath + ".json")
        print ("IR network structure is saved as [{}.json].".format(args.dstPath))

        prototxt = graph.as_model_def(info_model).SerializeToString()
        with open(args.dstPath + ".pb", 'wb') as of:
            of.write(prototxt)
        print ("IR network structure is saved as [{}.pb].".format(args.dstPath))

        import numpy as np
        with open(args.dstPath + ".npy", 'wb') as of:
            np.save(of, data)
        print ("IR weights are saved as [{}.npy].".format(args.dstPath))

        return 0

    elif args.src == 'keras':

        from common.keras.keras_converter import KerasConverter

        mxnet_model = KerasConverter((args.network, args.weights, info_model))
        mxnet_model.keras_to_IR()
        mxnet_model.save_to_json(args.dstPath + '.json')
        mxnet_model.save_to_proto(args.dstPath + '.pb')
        mxnet_model.save_weights(args.dstPath + '.npy')

        return 0

    elif args.src == 'tensorflow' or args.src == 'tf':
        assert args.network or args.weights
        if not args.network:
            if args.dstNodeName is None:
                raise ValueError("Need to provide the output node of Tensorflow model.")
            if args.inNodeName is None:
                raise ValueError("Need to provide the input node of Tensorflow model.")
            if inputshape is None:
                raise ValueError("Need to provide the input node shape of Tensorflow model.")
            assert len(args.inNodeName) == len(inputshape)
            # from mmdnn.conversion.tensorflow.tensorflow_frozenparser import TensorflowParser2
            from common.tensorflow.tensorflow_frozenparser import TensorflowParser2
            parser = TensorflowParser2(args.weights, inputshape, args.inNodeName, args.dstNodeName, info_model=info_model)

        else:
            # from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser
            from common.tensorflow.tensorflow_parser import TensorflowParser
            if args.inNodeName and inputshape[0]:
                parser = TensorflowParser(args.network, args.weights, args.dstNodeName, inputshape[0], args.inNodeName, info_model=info_model)
            else:
                parser = TensorflowParser(args.network, args.weights, args.dstNodeName, info_model=info_model)

        parser.run(args.dstPath)

        return 0

    elif args.src == 'mxnet':
        from common.mxnet.mxnet_converter import MxNetConverter

        args = ('IR', args.network, args.weights, inputshape[0], info_model)
        mxnet_model = MxNetConverter(args)
        mxnet_model.mxnet_to_IR()
        mxnet_model.save_to_json(args.dstPath + '.json')
        mxnet_model.save_to_proto(args.dstPath + '.pb')
        mxnet_model.save_weights(args.dstPath + '.npy')

        return 0

    elif args.src == 'pytorch':
        import common.pytorch as pt 
        # convert pytorch to IR
        IR_pth = args.dstPath
        parser = pt.PytorchParser(args.network, inputshape[0])
        parser.run(IR_pth)

        return 0

    else:
        raise ValueError("Unknown framework [{}].".format(args.src))

    return 0


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description = 'Convert other model file formats to IR format.')

    parser.add_argument(
        '--src', '-s',
        type=_text_type,
        choices=["caffe", "mxnet", "keras", "tensorflow", 'tf', 'pytorch'],
        help="Source toolkit name of the model to be converted.")

    parser.add_argument(
        '--weights', '-w',
        type=_text_type,
        default=None,
        help='Path to the model weights file of the external tool.')

    parser.add_argument(
        '--network', '-n', '-in',
        type=_text_type,
        default=None,
        help='Path to the model network file of the external tool.')

    parser.add_argument(
        '--dstPath', '-d',
        type=_text_type,
        required=True,
        help='Path to save the IR model.')

    # parser.add_argument(
    #     '--outPath', '-o',
    #     type=_text_type,
    #     required=True,
    #     help='Out of Path to save the IR model.')

    # Tensorflow
    parser.add_argument(
        '--inNodeName', '-inode',
        nargs='+',
        type=_text_type,
        default=None,
        help="[Tensorflow] Input nodes' name of the graph.")

    parser.add_argument(
        '--dstNodeName', '-node',
        nargs='+',
        type=_text_type,
        default=None,
        help="[Tensorflow] Output nodes' name of the graph.")

    parser.add_argument(
        '--inputShape',
        nargs='+',
        type=_text_type,
        default=None,
        help='[Tensorflow/MXNet] Input shape of model (channel, height, width)')


    # Caffe
    parser.add_argument(
        '--caffePhase',
        type=_text_type,
        default='TRAIN',
        help='[Caffe] Convert the specific phase of caffe model.')

    return parser

def _main():
    parser = _get_parser()
    args = parser.parse_args()
    ret = _convert(args)
    sys.exit(int(ret)) # cast to int or else the exit code is always 1


if __name__ == '__main__':
    _main()
