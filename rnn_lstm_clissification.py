import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb

tf.set_random_seed(1)

mnist = input_data.read_data_sets('./mnist', one_hot=True)

learning_rate = 0.001
train_iters = 100000
input_size = 28
time_step = 28
batch_size = 64
num_classes = 10
num_hidden_unit = 128


tf_x = tf.placeholder(tf.float32, [None, input_size * time_step])
tf_y = tf.placeholder(tf.float32, [None, num_classes])

weights = {
    # 28*128
    'in': tf.Variable(tf.random_normal([input_size, num_hidden_unit])),
    # 128*10
    'out': tf.Variable(tf.random_normal([num_hidden_unit, num_classes]))
}
biases = {
    # 128
    'in': tf.Variable(tf.constant(0.1, shape=[num_hidden_unit, ])),
    # 10
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes, ]))
}


def RNN(x, w, b):
    # cell input
    x = tf.reshape(x, [-1, input_size])
    x_in = tf.matmul(x, w['in']) + b['in']
    x_in = tf.reshape(x_in, [-1, time_step, num_hidden_unit])

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_unit, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    # cell output
    result = tf.matmul(states[1], w['out']) + b['out']
    return result


pred = RNN(tf_x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_pre = tf.equal(tf.argmax(pred, 1), tf.argmax(tf_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < train_iters:
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run([train_op], feed_dict={tf_x: x_batch, tf_y: y_batch})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={tf_x: x_batch, tf_y: y_batch}))
        step += 1
