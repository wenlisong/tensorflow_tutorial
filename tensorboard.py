import tensorflow as tf
import numpy as np


def add_layer(inputs, in_shape, out_shape, n_layer, activation_function):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_shape, out_shape]), name='W')
            tf.summary.histogram(layer_name + '/weights', weights)
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_shape]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_b'):
            wx_b = tf.matmul(inputs, weights) + biases

        if activation_function is None:
            outputs = wx_b
        else:
            outputs = activation_function(wx_b)

        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


if __name__ == '__main__':
    x_data = np.linspace(-1, 1, 10000, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        for step in range(1000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if step % 50 == 0:
                result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
                writer.add_summary(result, step)
