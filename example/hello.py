#!/usr/bin/env python3
# coding=utf-8

'''
Hello World，用来验证TensorFlow安装是否成功
TensorFlow官方给出的验证安装是否成功的例子
'''
# Import the library
import tensorflow as tf

# Define the graph
hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

'''
使用TensorFlow来进行简单运算
'''
a = tf.constant(10)
b = tf.constant(32)

compute_op = tf.add(a, b)

# Define the session to run graph
with tf.Session() as sess:
    print(sess.run(hello))
    print(sess.run(compute_op))
