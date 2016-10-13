import tensorflow as tf
import numpy as np

a = tf.constant(2)
b = tf.constant(3)

c = a+b

k = tf.constant([[2,3], [4,5]])
j = tf.constant([[2,3], [4,5]])

m = np.array([[1,2], [2,3]], dtype='float')
n = np.array([[3,4], [2,3]], dtype='float')

kj1 = k*j
kj2 = tf.matmul(k, j)

mn = tf.matmul(m, n)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

print(sess.run(c))
print(sess.run(kj1))

print(sess.run(kj1))
print(sess.run(kj2))

print(sess.run(mn))

