import tensorflow as tf
import numpy as np
import time

# generate training set
x_train = np.random.rand(100).astype(np.float32)
y_train = 0.3 * x_train + 0.1

# Network Model
weight = tf.Variable(tf.random_uniform([1], -1, 1))
bias = tf.Variable(tf.zeros([1]))

y = weight * x_train + bias

loss = tf.reduce_mean(tf.square(y - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
train = optimizer.minimize(loss)
# end

t0 = time.clock()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(weight), sess.run(bias))
for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print("step ", step, sess.run(weight), sess.run(bias))

t1 = time.clock()
print(t1 - t0)
