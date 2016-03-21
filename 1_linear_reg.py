import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

trX = np.linspace(-1, 1, 101)
trY = 2*trX + np.random.randn(*trX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X, w, b):
    return tf.mul(X, w) + b

w = tf.Variable(tf.random_normal([1]), name="weights")
b = tf.Variable(tf.random_normal([1]), name="bias")
y_model = model(X, w, b)

cost = tf.reduce_sum(tf.pow(Y-y_model, 2))/101

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()

init = tf.initialize_all_variables()

sess.run(init)


for i in range(100):
    for(x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})


print(sess.run([w, b]))
print(trY)
print(sess.run(y_model, feed_dict={X: trX}))

#plt.plot(trX, trY, 'ro')
#plt.plot(trX, y_model)
#plt.show()




