import os
import sys
sys.path.append('./../')
import google.protobuf.text_format as text_format
from six import text_type as _text_type


def _convert(args):
    if args.dst == 'caffe':
        from common.caffe.caffe_emitter import CaffeEmitter
        if args.IRWeightPath is None:
            emitter = CaffeEmitter(args.IRModelPath)
        else:
            assert args.dstWeightPath
            emitter = CaffeEmitter((args.IRModelPath, args.IRWeightPath))

        emitter.run(args.dstModelPath, args.dstWeightPath, args.phase)

    elif args.dst == 'tensorflow' or args.dst == 'tf':
        from common.tensorflow.tensorflow_emitter import TensorflowEmitter
        if args.IRWeightPath is None:
            # Convert network architecture only
            emitter = TensorflowEmitter(args.IRModelPath)
        else:
            emitter = TensorflowEmitter((args.IRModelPath, args.IRWeightPath))

        emitter.run(args.dstModelPath, args.dstWeightPath, args.phase)

    elif args.dst == 'pytorch':
        #from IR generate pytorch code 
        pytorch_code = os.path.join(args.dstModelPath, "code.py")
        pytorch_npy = os.path.join(args.dstModelPath, "weights.npy")
        recover = pt.PytorchEmitter((args.IRModelPath, args.IRWeightPath))
        recover.run(pytorch_code, pytorch_npy)

        #from pytorch code generate pytorch model
        pytorch_d_pth = os.path.join(args.dstModelPath, "pytorch_model.pth")
        name = "model"
        import imp
        MainModel = imp.load_source(name, pytorch_code)
        pt.save_model(MainModel, pytorch_code, pytorch_npy, pytorch_d_pth)

    elif args.dst == 'mxnet':
        from common.mxnet.mxnet_recover import MXNetRecover
        args = (args.IRModelPath, args.IRWeightPath, args.dstModelPath)
        mxnet_model = MXNetRecover(args)
        mxnet_model.IR_to_mxnet()
    else:
        assert False

    return 0


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description = 'Convert IR model file formats to other format.')

    parser.add_argument(
        '--phase',
        type=_text_type,
        choices=['train', 'test'],
        default='test',
        help='Convert phase (train/test) for destination toolkits.'
    )

    parser.add_argument(
        '--dst', '-f',
        type=_text_type,
        choices=['caffe', 'mxnet', 'tensorflow', 'tf', 'pytorch'],
        required=True,
        help='Format of model at srcModelPath (default is to auto-detect).')

    parser.add_argument(
        '--IRModelPath', '-n', '-in',
        type=_text_type,
        required=True,
        help='Path to the IR network structure file.')

    parser.add_argument(
        '--IRWeightPath', '-w', '-iw',
        type=_text_type,
        required=False,
        default=None,
        help = 'Path to the IR network structure file.')

    parser.add_argument(
        '--dstModelPath', '-d', '-o',
        type = _text_type,
        required = True,
        help = 'Path to save the destination model')

    # MXNet
    parser.add_argument(
        '--dstWeightPath', '-dw', '-ow',
        type=_text_type,
        default=None,
        help='[MXNet] Path to save the destination weight.')
    
    return parser

    # CUDA_VISIBLE_DEVICES=1 python IRToModel.py --phase test -f tf -n kit_imagenet.pb -w kit_imagenet.npy -d tf


def _main():
    parser=_get_parser()
    args = parser.parse_args()
    ret = _convert(args)
    sys.exit(int(ret)) # cast to int or else the exit code is always 1


if __name__ == '__main__':
    _main()