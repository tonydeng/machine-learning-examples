# TensorFlow 安装

[官方的macOS安装文档](https://www.tensorflow.org/install/install_mac)


使用pip3安装

```bash
pip3 install --upgrade tensorflow
```

安装过程

```bash
Collecting tensorflow
  Downloading tensorflow-1.0.0-cp36-cp36m-macosx_10_11_x86_64.whl (39.7MB)
    100% |████████████████████████████████| 39.7MB 33kB/s
Collecting numpy>=1.11.0 (from tensorflow)
  Downloading numpy-1.12.0-cp36-cp36m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl (4.4MB)
    100% |████████████████████████████████| 4.4MB 309kB/s
Collecting protobuf>=3.1.0 (from tensorflow)
  Downloading protobuf-3.2.0-py2.py3-none-any.whl (360kB)
    100% |████████████████████████████████| 368kB 2.7MB/s
Collecting six>=1.10.0 (from tensorflow)
  Using cached six-1.10.0-py2.py3-none-any.whl
Requirement already up-to-date: wheel>=0.26 in /usr/local/lib/python3.6/site-packages (from tensorflow)
Collecting setuptools (from protobuf>=3.1.0->tensorflow)
  Using cached setuptools-34.3.2-py2.py3-none-any.whl
Collecting appdirs>=1.4.0 (from setuptools->protobuf>=3.1.0->tensorflow)
  Using cached appdirs-1.4.3-py2.py3-none-any.whl
Collecting packaging>=16.8 (from setuptools->protobuf>=3.1.0->tensorflow)
  Using cached packaging-16.8-py2.py3-none-any.whl
Collecting pyparsing (from packaging>=16.8->setuptools->protobuf>=3.1.0->tensorflow)
  Using cached pyparsing-2.2.0-py2.py3-none-any.whl
Installing collected packages: numpy, six, appdirs, pyparsing, packaging, setuptools, protobuf, tensorflow
  Found existing installation: setuptools 32.2.0
    Uninstalling setuptools-32.2.0:
      Successfully uninstalled setuptools-32.2.0
Successfully installed appdirs-1.4.3 numpy-1.12.0 packaging-16.8 protobuf-3.2.0 pyparsing-2.2.0 setuptools-34.3.2 six-1.10.0 tensorflow-1.0.0
```

简单验证tensorflow安装是否成功，来一段简单的`Hello World`

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

输出结果

```bash
b'Hello, TensorFlow!'
```

查看代码 [hellp.py](demo/hello.py)
