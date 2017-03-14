#!/usr/bin/env python3
# coding=utf-8
'''
通过TensorFlow实现的随机梯度算法，在训练足够长的时机后可以自动求加函数中的斜率和截距。
'''
import tensorflow as tf
import numpy as np

# Prepare train data
# 随机生成100个点(x,y)
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
# 构建线性模型的tensor变量w,b
X = tf.placeholder('float')
Y = tf.placeholder('float')
w = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')

# 构建损失方程，优化器及训练模型操作train_op
loss = tf.square(Y - tf.multiply(X, w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    # 构建变量初始化操作init，并初始化所有TensorFlow变量
    sess.run(tf.global_variables_initializer())
    epoch = 1
    '''
    训练该线性模型，输出模型参数
    经过1000次的迭代，我们看到输出的斜率w约为2，截距b约为10，与我们构建的数据之间的关联关系十分吻合。
    '''
    for i in range(10):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X:x, Y:y})
            print("Epoch: {}, w: {}, b: {}".format(epoch,w_value,b_value))
            epoch +=1
