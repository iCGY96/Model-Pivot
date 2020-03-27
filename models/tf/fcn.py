import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

# def fcn(x, is_train):
#     net = x
#     with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                         trainable=is_train,
#                         activation_fn=tf.nn.relu,
#                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                         weights_regularizer=slim.l2_regularizer(5e-4),
#                         kernel_size=[3, 3],
#                         padding='SAME'):
#         with slim.arg_scope([slim.conv2d],
#                             stride=1,
#                             normalizer_fn=slim.batch_norm):
            
#             net = slim.repeat(net, 3, slim.conv2d, 16, scope='conv1')
#             net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 256
#             net = slim.repeat(net, 3, slim.conv2d, 32, scope='conv2')
#             net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 128
#             net = slim.repeat(net, 3, slim.conv2d, 64, scope='conv3')
#             net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 64
#             net = slim.repeat(net, 3, slim.conv2d, 256, scope='conv4')
#             net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 32
#             net = slim.repeat(net, 3, slim.conv2d, 1024, scope='conv5')
#             net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 16

#             net = slim.conv2d_transpose(net, num_outputs=512, stride=2, scope='deconv1')  # 32
#             net = slim.repeat(net, 3, slim.conv2d, 512, scope='conv6')
#             net = slim.conv2d_transpose(net, num_outputs=128, stride=2, scope='deconv2')  # 64
#             net = slim.repeat(net, 3, slim.conv2d, 128, scope='conv7')
#             net = slim.conv2d_transpose(net, num_outputs=32, stride=2, scope='deconv3')  # 128
#             net = slim.repeat(net, 3, slim.conv2d, 32, scope='conv8')
#             net = slim.conv2d(net, 2, scope='output', activation_fn=None, normalizer_fn=None)

#     return net

def FCN8_atonce(images, num_classes=3):
    paddings = tf.constant([[0, 0], [96, 96], [96, 96], [0, 0]])
    pad_images = tf.pad(images, paddings, 'CONSTANT')

    height = images.shape[1].value
    width = images.shape[2].value

    model = nets.vgg
    with slim.arg_scope(model.vgg_arg_scope()):
        score, end_points = model.vgg_16(pad_images, num_classes, spatial_squeeze=False)
    
    with tf.variable_scope('FCN'):
        score_pool3 = slim.conv2d(0.0001 * end_points['vgg_16/pool3'], num_classes, 1, scope='score_pool3')
        score_pool4 = slim.conv2d(0.01 * end_points['vgg_16/pool4'], num_classes, 1, scope='score_pool4')
    
        # score_pool3c =  tf.image.resize_images(score_pool3, (int(height / 8), int(width / 8)))
        # score_pool4c =  tf.image.resize_images(score_pool4, (int(height / 16), int(width / 16)))
        score_pool3c = tf.image.crop_to_bounding_box(score_pool3, 12, 12, int(height / 8), int(width / 8))
        score_pool4c = tf.image.crop_to_bounding_box(score_pool4, 6, 6, int(height / 16), int(width / 16))

        up_score = slim.conv2d_transpose(score, num_classes, 4, stride=2, scope='up_score')
        fuse1 = tf.add(up_score, score_pool4c, name='fuse1')

        up_fuse1 = slim.conv2d_transpose(fuse1, num_classes, 4, stride=2, scope='up_fuse1')
        fuse2 = tf.add(up_fuse1, score_pool3c, name='fuse2')

        up_fuse2 = slim.conv2d_transpose(fuse2, num_classes, 16, stride=8, scope='up_fuse2')

        pred = tf.argmax(up_fuse2, 3, name='pred')

    return tf.expand_dims(pred, 3), up_fuse2


def test(x, model_path):
    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3])
    _, logits = FCN8_atonce(data_input)
    labels = tf.identity(logits, name='OX_output')

    init_fn = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_fn)
        saver = tf.train.Saver()
        preds = sess.run([labels], feed_dict = {data_input : x})
        saver.save(sess, model_path + '/tensorflow/fcn/fcn.ckpt')
    
    return preds
