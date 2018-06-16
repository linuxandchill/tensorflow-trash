import tensorflow as tf

x = tf.constant(6)
y = tf.constant(7)

result = tf.multiply(x,y)

with tf.Session() as sess:
    print(sess.run(result))
