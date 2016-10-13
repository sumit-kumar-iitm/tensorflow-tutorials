import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/data_logistics/data_with_labels.npz')

normalized_train_data = data['arr_0']/255.
labels = data['arr_1']

def to_onehot(labels, classes=5):
    outputlabels = np.zeros((len(labels), 5))
    for i, l in enumerate(labels):
        outputlabels[i, l] = 1
    return outputlabels

onehot_labels = to_onehot(labels, 5)

indices = np.random.permutation(normalized_train_data.shape[0])
valid_cnt = int(normalized_train_data.shape[0]*0.1)
test_idx, train_idx = indices[:valid_cnt], indices[valid_cnt:]

test, train = normalized_train_data[test_idx, :], normalized_train_data[train_idx, :]
onehot_test_labels, onehot_train_labels = onehot_labels[test_idx, :], onehot_labels[train_idx, :]

test_flattened = test.reshape([-1, 1296])
train_flattened = train.reshape([-1, 1296])

x = tf.placeholder('float', [None, 1296])
y_ = tf.placeholder('float', [None, 5])

W = tf.Variable(tf.zeros([1296, 5]))
b = tf.Variable(tf.zeros([5]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y + 1e-50, y_))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#training
sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 1000
test_acc = np.zeros(epochs//10)
train_acc = np.zeros(epochs//10)
for i in range(epochs):
    if i%10 == 0:
        A = sess.run(accuracy, feed_dict={x: train_flattened, y_: onehot_train_labels})
        train_acc[i//10] = A
        A = sess.run(accuracy, feed_dict={x: test_flattened, y_: onehot_test_labels})
        test_acc[i//10] = A
    sess.run(train_step, feed_dict={x: train_flattened, y_: onehot_train_labels})

print test_acc[-1]
print train_acc[-1]



