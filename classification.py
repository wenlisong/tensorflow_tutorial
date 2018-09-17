import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)


def add_layer(inputs, in_shape, out_shape, activation_function):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_shape, out_shape]), name='W')
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_shape]) + 0.1, name='b')
        with tf.name_scope('Wx_b'):
            wx_b = tf.matmul(inputs, weights) + biases

        if activation_function is None:
            outputs = wx_b
        else:
            outputs = activation_function(wx_b)

        return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    res = sess.run(accuracy, feed_dict={xs: v_xs})
    return res


if __name__ == '__main__':
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 28x28 pixel
        ys = tf.placeholder(tf.float32, [None, 10], name='y_input')  # 10 classes

    # add output layer
    l1 = add_layer(xs, 784, 300, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l1, 300, 10, activation_function=tf.nn.softmax)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        for step in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            if step % 100 == 0:
                print(compute_accuracy(mnist.test.images, mnist.test.labels))
                result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
                writer.add_summary(result, step)
