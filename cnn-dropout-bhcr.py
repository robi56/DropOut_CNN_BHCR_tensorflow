import  numpy as np
import  pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from data import read_dataset

mnist = read_dataset("Data")


import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([784,50]))
b = tf.Variable(tf.zeros([50]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
result = list()
mini_result = list()
a1 = []
b1 = []

for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        mini_result = list()
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        a1.append(i)
        b1.append(train_accuracy)

        mini_result.append(i)
        mini_result.append(train_accuracy)
        result.append(mini_result)
        #result.append(mini_result)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


thefile = open('cnn_e10000_b100.txt', 'w')

thefile.write("%s\n" % result)

for item in result:
    thefile.write("%s\n" % item)



a2 = []
b2 = []



for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        mini_result = list()
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        a2.append(i)
        b2.append(train_accuracy)

        mini_result.append(i)
        mini_result.append(train_accuracy)
        result.append(mini_result)
        #result.append(mini_result)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#print(result)
#plt.plot(a, b, 'ro')
#plt.axis([0, 6, 0, 20])
plt.show()

thefile = open('cnn_e10000_b50.txt', 'w')

thefile.write("%s\n" % result)

for item in result:
    thefile.write("%s\n" % item)



# plot with various axes scales
plt.figure(1)

# linear
plt.subplot(221)
plt.plot(a1, b1)
plt.yscale('epoch')
plt.title('batchsize')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(a2, b2)
plt.yscale('epoch')
plt.title('batchsize')
plt.grid(True)




plt.show()