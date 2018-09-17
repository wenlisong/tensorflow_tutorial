import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_state = tf.add(state, one)
update = tf.assign(state, new_state)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
