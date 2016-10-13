import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

try:
    from tqdm import tqdm
except:
    def tqdm(x, *args, **kwargs):
        return x

np.random.seed(0)

#Load data
data = np.load('data/data_logistics/data_with_labels.npz')
train_data = data['arr_0']/255
labels = data['arr_1']

print(train_data[0])
print labels[0]

plt.figure(figsize=(6,6))
f, plts = plt.subplots(5, sharex=True)

c = 91
for i in range(5):
    plts[i].pcolor(train_data[c + i*558], cmap=plt.cm.gray_r)


def to_onehot(labels, nclasses=5):
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i, l] = 1

    return outlabels

one_hot_labels = to_onehot(labels, 5)

indices = np.random.permutation(train_data.shape[0])

valid_cnt = int(train_data.shape[0] * 0.1)
test_idx, training_idx = indices[:valid_cnt], indices[valid_cnt:]

test, train = train_data[test_idx, :], train_data[training_idx, :]

onehot_test_labels, pnehot_train_labels = one_hot_labels[test_idx, :], one_hot_labels[training_idx, :]


sess = tf.Session()

#shape is 36*36
x = tf.placeholder("float", [None, 1296])
y_ = tf.placeholder("float", [None, 5])

#Variables
W = tf.Variable(tf.zeros([1296, 5]))
b = tf.Variable(tf.zeros([5]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y + 1e-50, y_))

#Training
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

