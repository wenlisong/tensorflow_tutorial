import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x.shape)
y = np.power(x, 2) + noise


def save():
    print('This is save')

    tf_x = tf.placeholder(tf.float32, x.shape)
    tf_y = tf.placeholder(tf.float32, y.shape)
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
    l2 = tf.layers.dense(l1, 1)
    loss = tf.losses.mean_squared_error(tf_y, l2)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for step in range(100):
            sess.run(train_op, {tf_x: x, tf_y: y})

        saver.save(sess, './params/', write_meta_graph=False)

        # plot data
        pred, l = sess.run([l2, loss], {tf_x: x, tf_y: y})
        plt.figure(1, figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(-1, 1.2, 'Save Loss = %.4f' % l, fontdict={'size': 15, 'color': 'red'})


def reload():
    print('This is reload')
    # build entire net again and restore
    tf_x = tf.placeholder(tf.float32, x.shape)  # input x
    tf_y = tf.placeholder(tf.float32, y.shape)  # input y
    l1_ = tf.layers.dense(tf_x, 10, tf.nn.relu)  # hidden layer
    l2_ = tf.layers.dense(l1_, 1)  # output layer
    loss_ = tf.losses.mean_squared_error(tf_y, l2_)  # compute cost

    sess = tf.Session()
    # don't need to initialize variables, just restoring trained variables
    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, './params/')

    # plotting
    pred, l = sess.run([l2_, loss_], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Reload Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()


if __name__ == '__main__':
    save()
    tf.reset_default_graph()
    reload()
