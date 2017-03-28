#!/usr/bin/env python3
# coding=utf-8
from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
    '''
    inX 用于分类的输入向量
    dataSet 输入的训练样本集
    labels 标签向量
    k 表示用于选择最近邻居的数目
    其中标签向量的元素数目和矩阵dataSet的行数相同
    '''
    dataSetSize = dataSet.shape[0]
    '''
    计算距离
    当前使用欧氏距离公式，技术两个向量点xA和xB之间的距离
    相关欧氏距离可以查看文档
    https://github.com/tonydeng/machine-learning-examples/blob/master/docs/math/euclidean-distance.md
    '''
    diffMat = tile(inX, (dataSetSize ,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()

    classCount={}
    '''
    选择距离最小的K个点
    '''
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1

    '''
    排序
    '''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse = True)

    return sortedClassCount[0][0]

def createDataSet():
    '''
    构造训练数据集
    '''
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def main():
    group, labels = createDataSet()
    # 结果应该是B
    print(classify0([0,0], group, labels, 3))


if __name__ == '__main__':
    main()
