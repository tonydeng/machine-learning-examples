#!/usr/bin/env python3
# coding=utf-8

'''
Hello World，用来验证TensorFlow安装是否成功
TensorFlow官方给出的验证安装是否成功的例子
'''

import tensorflow as tf

hello = tf.constant('Hello, World!')
sess = tf.Session()
print(sess.run(hello))
