import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def add_layer(inputs, in_shape, out_shape, activation_function):
    weight = tf.Variable(tf.random_normal([in_shape, out_shape]))
    bias = tf.Variable(tf.zeros([1, out_shape]) + 0.1)
    wx_b = tf.matmul(inputs, weight) + bias

    if activation_function is None:
        outputs = wx_b
    else:
        outputs = activation_function(wx_b)

    return outputs


if __name__ == '__main__':
    x_data = np.linspace(-1, 1, 10000, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10, tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    t1 = time.clock()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()
        while True:
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            loss_val = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            if step % 100 == 0:
                print("Step {0} : loss = {1}".format(step, loss_val))
                prediction_val = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, prediction_val, 'r-', linewidth=2)
                plt.pause(0.1)
                ax.lines.remove(lines[0])
            if loss_val < 0.001:
                break
            step += 1
    t2 = time.clock()
    print('Total time cost: {0}'.format(t2 - t1))
