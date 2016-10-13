import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/data_poly_reg/group5_train.txt', header=None, delim_whitespace = True, dtype = np.float32)  #instead of delim, we use sep=' '
test_data = pd.read_csv('data/data_poly_reg/group5_test.txt', header=None, delim_whitespace = True, dtype = np.float32)
plt.plot(data[0], data[1], 'ro')
plt.show()

n_observations = 1500

X = tf.placeholder("float")
Y = tf.placeholder("float")

Y_pred = tf.Variable(tf.random_normal([1]))
W = tf.Variable(tf.random_normal([1]))
for i in range(1, 6):
    W = tf.Variable(tf.random_normal([1]))
    Y_pred = tf.add(tf.mul(tf.pow(X, i), W), Y_pred)

#cost function
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

n_epochs = 1000

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    prev_training_cost = 0.0
    training_cost = 0.0
    for epoch in range(n_epochs):
        for(x, y) in zip(data[0], data[1]):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_cost = sess.run(cost, feed_dict={X: data[0], Y: data[1]})
        print(training_cost)
        if np.abs(training_cost - prev_training_cost) < 0.00001:
            break
        prev_training_cost = training_cost

    print("Predicted Values")
    #plt.plot(test_data[0], test_data[1], 'ro')
    #plt.show()
    plt.plot(data[0], sess.run(Y_pred, feed_dict={X: data[0]}), 'ro')
    plt.show()
    #print(np.abs(sess.run(Y_pred, feed_dict={X: test_data[0]}) - test_data[1]))

