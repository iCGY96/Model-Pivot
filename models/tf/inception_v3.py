import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.contrib.slim.nets import inception

model_map = {
    'inception_v3' : {
        'builder'     : lambda : inception.inception_v3,
        'arg_scope'   : inception.inception_v3_arg_scope,
        'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
        'num_classes' : 1001,
    }
}
slim = tf.contrib.slim

def test(x, model_path):

    model = 'inception_v3'
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
        saver.save(sess, model_path + '/tensorflow/inception_v3/inception_v3.ckpt')
    
    return preds