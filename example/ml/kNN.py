#!/usr/bin/env python3
# coding=utf-8
from numpy import *
import operator
import os
import matplotlib
import matplotlib.pyplot as plt

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

def file2matrix(filename):
    '''
    从文本文件中解析数据
    '''
    fr = open(filename)
    # 得到文件行数
    numberOfLines = len(fr.readlines())
    # 创建矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []

    fr = open(filename)
    index = 0
    # 解析数据到矩阵
    for line in fr.readlines():
        line = line.strip()
        listFromline = line.split('\t')
        returnMat[index,:] = listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index +=1

    return returnMat,classLabelVector

def plotshow(matrix, labels):
    '''
    使用matplotlib制作散点图
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 使用matrix的第二、三列数据
    ax.scatter(matrix[:,1], matrix[:,2], 15.0*array(labels), 15.0*array(labels))
    plt.show()

def autoNorm(dataSet):
    '''
    归一化数值
    在处理不同取值范围的特征值时，通常采用的方式是将数值归一化，如将取值范围处理为0到1或者-1到1之间
    下面的公式可以将任意取值范围的特征值转化为0到1区间的值
    newValue = (oldValue - min) / (max - min)
    '''
    # 将每列的最小值放在变量minVals,dataSet.min(0)中的参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值
    minVals = dataSet.min(0)
    # 将每列的最大值放在变量maxVals
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 特征值相除，使用tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRation = 0.10
    # 读取数据并将其转换为归一化特征值
    datingDataMat, datingLabels = file2matrix('docs/data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 计算测试向量的数量，决定了哪些数据用于测试
    m = normMat.shape[0]
    numTestVecs = int(m * hoRation)

    errorCount = 0.0
    for i in range(numTestVecs):
        # 将这两部分数据输入到原始kNN分类器函数classify0。
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

        if (classifierResult != datingLabels[i]): errorCount += 1.0

    # 输出结果
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('docs/data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels,3)
    print("You will peobably like thes person:", resultList[classifierResult - 1])

def main():
    # group, labels = createDataSet()
    # 结果应该是B
    # print(classify0([0,0], group, labels, 3))
    # datingDataMat, datingLabels = file2matrix('docs/data/datingTestSet2.txt')
    # print(datingDataMat)
    # print(datingLabels)
    # plotshow(datingDataMat, datingLabels)
    # datingClassTest()
    classifyPerson()


if __name__ == '__main__':
    main()
