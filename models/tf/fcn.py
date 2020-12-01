import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def FCN8_atonce(images, num_classes=3):
    paddings = tf.constant([[0, 0], [96, 96], [96, 96], [0, 0]])
    pad_images = tf.pad(images, paddings, 'CONSTANT')

    model = nets.vgg
    with slim.arg_scope(model.vgg_arg_scope()):
        score, end_points = model.vgg_16(pad_images, num_classes, spatial_squeeze=False, is_training=False)
    
    with tf.variable_scope('FCN'):
        score_pool3 = slim.conv2d(0.0001 * end_points['vgg_16/pool3'], num_classes, 1, scope='score_pool3')
        score_pool4 = slim.conv2d(0.01 * end_points['vgg_16/pool4'], num_classes, 1, scope='score_pool4')
    
        score_pool3c = tf.image.crop_to_bounding_box(score_pool3, 12, 12, 28, 28)
        score_pool4c = tf.image.crop_to_bounding_box(score_pool4, 6, 6, 14, 14)

        up_score = slim.conv2d_transpose(score, num_classes, 4, stride=2, scope='up_score', activation_fn=None)
        fuse1 = tf.add(up_score, score_pool4c, name='fuse1')

        up_fuse1 = slim.conv2d_transpose(fuse1, num_classes, 4, stride=2, scope='up_fuse1', activation_fn=None)
        fuse2 = tf.add(up_fuse1, score_pool3c, name='fuse2')

        # preds = slim.conv2d_transpose(fuse2, num_classes, 4, stride=2, scope='up_fuse2', activation_fn=None)
        y = slim.conv2d_transpose(fuse2, num_classes, 16, stride=8, scope='output', activation_fn=None)

    return y


def test(x, model_path):
    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3])
    logits = FCN8_atonce(data_input)
    labels = tf.identity(logits, name='OX_output')

    init_fn = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_fn)
        # saver.restore(sess, model_path + '/tensorflow/fcn/fcn.ckpt')
        preds = sess.run([labels], feed_dict = {data_input : x})
        saver.save(sess, model_path + '/tensorflow/fcn/fcn.ckpt')
    
    return preds
