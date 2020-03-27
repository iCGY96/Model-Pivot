import os
import sys
import imp
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Test for Torch
def torch(test_models, model_path, img_path):
    results_o, results_d = {}, {}

    from PIL import Image
    import torch
    import torchvision.models as models
    from torchvision import transforms
    from torch.autograd import Variable

    # Torch to IR
    from ox.pytorch.pytorch_parser import PytorchParser

    for model in test_models:
        if 'inception' in model: image_size = 299
        else: image_size = 224

        image = Image.open(img_path)
        transformation = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image_tensor = transformation(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        x = Variable(image_tensor)
        inputshape = [3, image_size, image_size]

        arch_filename = os.path.join(model_path, 'PyTorch', model+'.pth')

        # test model
        if 'resnet50' in model:
            model_eval = models.resnet50()
        elif 'inception' in model:
            from models.torch import inception
            model_eval = inception.inceptionresnetv2(pretrained=False)
        elif 'shufflenet' in model:
            from models.torch import shufflenet
            model_eval = shufflenet.shufflenet()
        elif 'fcn' in model:
            from models.torch import fcn
            model_eval = fcn.FCNs()

        model_eval.eval()
        predict = model_eval(x).data.numpy()
        preds = np.squeeze(predict)
        print('\033[1;31;40m')
        print(' Result of', model, ': ', np.argmax(preds))
        print('\033[0m')
        results_o[model] = preds
        torch.save(model_eval, arch_filename)

        # convert
        IR_filename = os.path.join(model_path, 'IR', model+'_torch')
        parser = PytorchParser(arch_filename, inputshape)
        parser.run(IR_filename)
    del parser
    del PytorchParser

    # IR to Torch
    from ox.pytorch.pytorch_emitter import PytorchEmitter

    for model in test_models:
        if 'inception' in model: image_size = 299
        else: image_size = 224

        image = Image.open(img_path)
        transformation = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image_tensor = transformation(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        x = Variable(image_tensor)
        inputshape = [3, image_size, image_size]

        arch_filename = os.path.join(model_path, 'IR', model+'_torch.pb')
        weight_filename = os.path.join(model_path, 'IR', model+'_torch.npy')
        converted_file = os.path.join(model_path, 'PyTorch', model+'_ox')
        emitter = PytorchEmitter((arch_filename, weight_filename))
        emitter.run(converted_file + '.py', converted_file + '.npy', 'test')

        model_converted = imp.load_source('PytorchModel', converted_file + '.py').KitModel(converted_file + '.npy')

        model_converted.eval()

        predict = model_converted(x).data.numpy()
        preds = np.squeeze(predict)

        print('\033[1;31;40m')
        print(' Result of ', model+'_ox : ', np.argmax(preds))
        print('\033[0m')
        results_d[model] = np.mean(results_o[model] - preds)
    del emitter
    del PytorchEmitter

    return results_d


# Test for Tensorflow
def tensorflow(test_models, model_path, img_path):
    results_o, results_d = {}, {}

    import tensorflow as tf
    from PIL import Image
    image = Image.open(img_path)

    # Tensorflow to IR
    from ox.tensorflow.tensorflow_parser import TensorflowParser

    for model in test_models:
        arch_filename = os.path.join(model_path, 'tensorflow', model, model+'.ckpt.meta')
        weight_filename = os.path.join(model_path, 'tensorflow', model, model+'.ckpt')
        
        # test model
        if 'resnet50' in model:
            img = np.array(image.resize((299, 299), Image.ANTIALIAS))
            x = np.expand_dims(img, axis=0)
            from models.tf import resnet50
            preds = resnet50.test(x, model_path)
        elif 'inception' in model:
            img = np.array(image.resize((224, 224), Image.ANTIALIAS))
            x = np.expand_dims(img, axis=0)
            from models.tf import inception_v3
            preds = inception_v3.test(x, model_path)
        elif 'shufflenet' in model:
            img = np.array(image.resize((224, 224), Image.ANTIALIAS))
            x = np.expand_dims(img, axis=0)
            from models.tf import shufflenet
            preds = shufflenet.test(x, model_path)
        elif 'fcn' in model:
            img = np.array(image.resize((224, 224), Image.ANTIALIAS))
            x = np.expand_dims(img, axis=0)
            from models.tf import fcn
            preds = fcn.test(x, model_path)
        
        print('\033[1;31;40m')
        print(' Result of', model, ': ', np.argmax(preds))
        print('\033[0m')
        results_o[model] = preds

        import tensorflow.contrib.keras as keras
        keras.backend.clear_session()

        # parser
        IR_filename = os.path.join(model_path, 'IR', model+'_tf')
        parser = TensorflowParser(arch_filename, weight_filename, ["OX_output"])
        parser.run(IR_filename)
    del parser
    del TensorflowParser

    # IR to Tensorflow
    from ox.tensorflow.tensorflow_emitter import TensorflowEmitter

    for model in test_models:
        arch_filename = os.path.join(model_path, 'IR', model+'_tf.pb')
        weight_filename = os.path.join(model_path, 'IR', model+'_tf.npy')
        converted_file = os.path.join(model_path, 'tensorflow', model, model+'_ox')

        emitter = TensorflowEmitter((arch_filename, weight_filename))
        emitter.run(converted_file + '.py', None, 'test')

        # test model
        if 'resnet' in model:
            img = image.resize((299, 299), Image.ANTIALIAS)
        else:
            img = image.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        x = np.expand_dims(img, axis=0)

        model_converted = imp.load_source('TFModel', converted_file + '.py').KitModel(weight_filename)

        input_tf, model_tf = model_converted

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            predict = sess.run(model_tf, feed_dict = {input_tf : x})
        del model_converted
        del sys.modules['TFModel']
        preds = np.squeeze(predict)

        print('\033[1;31;40m')
        print(' Result of ', model+'_ox : ', np.argmax(preds))
        print('\033[0m')
        results_d[model] = np.mean(results_o[model] - preds)

    del emitter
    del TensorflowEmitter

    return np.array(results_d)

def mk_dirs(path):
    if not os.path.exists(path):
        os.mkdirs(path)
        return True
    return False

if __name__=='__main__':
    # ['resnet50', 'inception_v3', 'shufflenet', 'fcn']
    test_models = ['resnet50', 'inception_v3', 'shufflenet', 'fcn']
    model_path = './../models'
    img_path = os.path.join('./', 'elephant.jpg')

    # mkdirs
    mk_dirs(os.path.join(model_path, 'IR'))
    mk_dirs(os.path.join(model_path, 'tensorflow'))
    mk_dirs(os.path.join(model_path, 'PyTorch'))

    r_tf_d = tensorflow(test_models, model_path, img_path)
    r_torch_d = torch(test_models, model_path, img_path)

    print('tf error: \n', r_tf_d)
    print('torch error: \n', r_torch_d)