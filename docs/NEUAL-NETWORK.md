# 关于神经网络的简单介绍

## 什么是神经网络？

人脑总共有超过千亿个神经云细胞，通过神经突触相互连接。如果一个神经元被足够的输入所激活，那么它也会激活其他神经元，这个过程就叫做思考。

我们可以在计算机上创建神经网络，来对这个过程进行建模，且并不需要模拟分子级的生物复杂性，只要观其大略即可。

## 模拟例子

为了简化起见，我们只模拟一个神经元，含有三个输入和一个输出。

![nerve cell](https://jizhi-10061919.file.myqcloud.com/blog/cafd4d2eeb64ddddc913f7f3fdd16727.png)

我们将训练这个神经元来解决下面这个问题。

前四个样本叫做“训练集”，你能够求出模式吗？ `?`应该是`0`还是`1`呢？

![training set](https://jizhi-10061919.file.myqcloud.com/blog/fa1856f4db09ca057549b1a592be5bcd.png)

或许已经发现了，输出总是与第一列的输入相等，所以 `?` 应该是 `1`。


## 训练过程

问题虽然简单，但是如何教会神经元来正确的回答这个问题呢？

我们要给每一个输入赋予一个权重，权重可能为正也可能为负。

权重的绝对值，代表了输入对输出的决定权。

在开始之前，我们先将权重设置为随机数，再开始训练过程。

1. 从训练集样本读取输入，根据权重进行整理，再代入某个特殊的方程计算神经元的输出。
1. 计算误差，也就是神经元的实际输出和训练样本的期望输出之差。
1. 根据误差的方向，微调权重。
1. 重复10000次。

![train](https://jizhi-10061919.file.myqcloud.com/blog/f65f4fefd7abd0e6d18340a17f0016f9.jpeg)

最终神经元的权重会达到训练集的最优值。

如果我们让神经元去思考一个新的形势，遵循相同的过程，应该会得到一个不错的预测。

## 计算神经元输出的方程

你可能好奇，计算神经元输出的“特殊方程”是什么？

首先，我们取神经元输入的加权总和：

> ∑weighti⋅inputi=weight1⋅input1+weight2⋅input2+weight3⋅in

接下来，我们进行正规化，将结果限制在0和1之间。这里用到一个很方便的函数，叫[`Sigmoid`函数](https://zh.wikipedia.org/wiki/S%E5%87%BD%E6%95%B0)：

Sigmoid函数得名因其形状像S字母。

![S函数](https://wikimedia.org/api/rest_v1/media/math/render/svg/a26a3fa3cbb41a3abfe4c7ff88d47f0181489d13)

Sigmoid函数的级数展开后的表示

![S函数](https://wikimedia.org/api/rest_v1/media/math/render/svg/d452330986d5f394c2a5af528efbaa028bcec3e3)

Sigmoid函数是一个生物学中常见的S型的函数，也称为S型生长曲线。

![Sigmoid](https://jizhi-10061919.file.myqcloud.com/blog/c0d8be2c18c551eefa1f3f2fb4f0b3e6.png)

由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间。

## 调整权重的方程

在训练进程中，我们需要调整权重，但是具体如何调整呢？就要用到“误差加权导数”方程：

> Adjustweihtsby = error.input.SigmoidCurveGradient(output)

为什么是这个方程？首先，我们希望调整量与误差量成正比，然后在乘以输入(0-1)。如果输入为0，那么权重就不会被调整。最后乘以Sigmoid曲线的梯度，便于理解，请考虑：

1. 我们使用Sigmoid曲线计算神经元输出。
1. 如果输出绝对值很大，就标识该神经元是很确定的。（有正反两种可能）。
1. Sigmoid曲线在绝对值较大处的梯度较小。
1. 如果神经元确信当前权重值是正确的，那么就不需要太大调整。乘以Sigmoid曲线梯度可以实现。

Sigmoid曲线的梯度可以有导数获得：

> SigmoidCurveGradient(output) = output.(1 - output)

代入公式可以获得最终的权重调整方程：

> Adjustweihtsby = error.input.output.(1 - output)

实际上也有其他让神经元学习更快的方程，这里主要是取其相对简单的优势。

## Python代码实现

没有使用神经网络库，使用了numpy的几个函数, `array`(创建矩阵), `dot`(矩阵乘法), `random`(随机数), `exp`(自然对常数)

具体代码，可以查看[neual-network.py](../example/neaal/neual-network.py)
