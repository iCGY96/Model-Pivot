import numpy as np
import tensorflow as tf

BATCH_SIZE = 1
LSTM_UNITS = 128


def test(x, model_path):
    x = np.reshape(x, (-1, 224 * 224 * 3))
    with tf.variable_scope('LSTM'):
        data_input = tf.placeholder(name='input',
                                    dtype=tf.float32,
                                    shape=[None, 224 * 224 * 3])

        cell = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS, state_is_tuple=True)
        init_state = cell.zero_state(BATCH_SIZE, tf.float32)
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
            cell, [data_input], initial_state=init_state)
    labels = tf.squeeze(rnn_outputs, name='OX_output')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        preds = sess.run(rnn_outputs, feed_dict={data_input: x})
        saver.save(sess, model_path + '/tensorflow/lstm/lstm.ckpt')

    return preds
