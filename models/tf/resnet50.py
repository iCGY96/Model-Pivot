import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.contrib.slim.nets import resnet_v2

model_map = {
    'resnet50' : {
        'builder'     : lambda : resnet_v2.resnet_v2_50,
        'arg_scope'   : resnet_v2.resnet_arg_scope,
        'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
        'num_classes' : 1001,
    }
}
slim = tf.contrib.slim

def test(x, model_path):

    model = 'resnet50'
    with slim.arg_scope(model_map[model]['arg_scope']()):
        data_input = model_map[model]['input']()
        logits, endpoints = model_map[model]['builder']()(
            data_input,
            num_classes=model_map[model]['num_classes'],
            is_training=False)
        labels = tf.squeeze(logits, name='OX_output')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        preds = sess.run(labels, feed_dict = {data_input : x})
        saver.save(sess, model_path + '/tensorflow/resnet50/resnet50.ckpt')
    
    return preds