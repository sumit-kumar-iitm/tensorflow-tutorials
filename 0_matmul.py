import tensorflow as tf
import numpy as np

#creating placeholders for matrices a and b of float type
a = tf.placeholder("float")
b = tf.placeholder("float")

#placeholder to
result = tf.matmul(a, b)

sess = tf.Session()

a_ = np.array([[1, 2], [2, 4]])
b_ = np.array([[2, 1], [1, 2]])

print(sess.run(result, feed_dict={a: a_, b: b_}))
