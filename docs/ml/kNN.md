# k-NN算法

在模式识别领域中，**最近邻居法（KNN算法，又译K-近邻算法）** 是一种用于分类和回归的非参数统计方法。在这两种情况下，输入包含特征空间中的k个最接近的训练样本。

* 在`k-NN`分类中，输出是一个分类族群。一个对象的分类是由其邻居的“多数表决”确定的，k个最近邻居（k为正整数，通常较小）中最常见的分类决定了赋予该对象的类别。若`k = 1`，则该对象的类别直接由最近的一个节点赋予。

* 在`k-NN`回归中，输出是该对象的属性值。该值是其k个最近邻居的值的平均值。

`最近邻居法`采用`向量空间模型`来分类，概念为相同类别的案例，彼此的相似度高，而可以借由计算与已知类别案例之相似度，来评估未知类别案例可能的分类。

K-NN是一种基于实例的学习，或者是局部近似和将所有计算推迟到分类之后的惰性学习。`k-近邻算法`是所有的机器学习算法中最简单的之一。

无论是分类还是回归，衡量邻居的权重都非常有用，使较近邻居的权重比较远邻居的权重大。例如，一种常见的加权方案是给每个邻居权重赋值为1/ d，其中d是到邻居的距离。

邻居都取自一组已经正确分类（在回归的情况下，指属性值正确）的对象。虽然没要求明确的训练步骤，但这也可以当作是此算法的一个训练样本集。

`k-近邻算法`的缺点是对数据的局部结构非常敏感。本算法与`K-平均算法（另一流行的机器学习技术）`没有任何关系，请勿与之混淆

## 特点

k-NN算法是用来做分类的，也就是说，有一个样本空间里的样本分成很几个类型，然后，给定一个待分类的数据，通过计算接近自己最近的`K个样本`来判断这个待分类数据属于哪个分类。**你可以简单的理解为由那离自己最近的K个点来投票决定待分类数据归为哪一类**

![kNN算法示意图](../images/kNN/KnnClassification.png)

从上图中我们可以看到，图中的有两个类型的样本数据，一类是蓝色的正方形，另一类是红色的三角形。而那个绿色的圆形是我们待分类的数据。

1. 如果K=3，那么离绿色点最近的有2个红色三角形和1个蓝色的正方形，这3个点投票，于是绿色的这个待分类点属于红色的三角形。

1. 如果K=5，那么离绿色点最近的有2个红色三角形和3个蓝色的正方形，这5个点投票，于是绿色的这个待分类点属于蓝色的正方形。

我们可以看到，机器学习的本质——是基于一种数据统计的方法！

## 机器学习与数据挖掘-K最近邻(KNN)算法的实现（java和python版）

> 来源：http://blog.csdn.net/u011067360/article/details/45937327

KNN算法基础思想前面文章可以参考，这里主要讲解java和python的两种简单实现，也主要是理解简单的思想。

### python版本

这里实现一个手写识别算法，这里只简单识别0~9熟悉，在上篇文章中也展示了手写识别的应用，可以参考：[机器学习与数据挖掘-logistic回归及手写识别实例的实现](http://blog.csdn.net/u011067360/article/details/45624517)


输入：每个手写数字已经事先处理成32*32的二进制文本，存储为txt文件。0～9每个数字都有10个训练样本，5个测试样本。训练样本集如下图：左边是文件目录，右边是其中一个文件打开显示的结果，看着像1，这里有0~9，每个数字都有是个样本来作为训练集。

![训练集](https://wizardforcel.gitbooks.io/dm-algo-top10/content/img/20150523211334473.jpg)

第一步：将每个txt文本转化为一个向量，即3232的数组转化为11024的数组，这个1*1024的数组用机器学习的术语来说就是特征向量。

```python
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```

第二步：训练样本中有1010个图片，可以合并成一个1001024的矩阵，每一行对应一个图片，也就是一个txt文档。

```python
def handwritingClassTest():

    hwLabels = []
    trainingFileList = listdir('trainingDigits')  
    print trainingFileList        
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]          
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #print hwLabels
        #print fileNameStr   
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        #print trainingMat[i,:]
        #print len(trainingMat[i,:])

    testFileList = listdir('testDigits')       
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
```

第三步：测试样本中有105个图片，同样的，对于测试图片，将其转化为11024的向量，然后计算它与训练样本中各个图片的“距离”（这里两个向量的距离采用欧式距离），然后对距离排序，选出较小的前k个，因为这k个样本来自训练集，是已知其代表的数字的，所以被测试图片所代表的数字就可以确定为这k个中出现次数最多的那个数字。

```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #tile(A,(m,n))   
    print dataSet
    print "----------------"
    print tile(inX, (dataSetSize,1))
    print "----------------"
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      
    print diffMat
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                  
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()            
    classCount={}                                      
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

全部实现代码：

```python
#-*-coding:utf-8-*-
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #tile(A,(m,n))   
    print dataSet
    print "----------------"
    print tile(inX, (dataSetSize,1))
    print "----------------"
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      
    print diffMat
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                  
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()            
    classCount={}                                      
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():

    hwLabels = []
    trainingFileList = listdir('trainingDigits')  
    print trainingFileList        
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]          
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #print hwLabels
        #print fileNameStr   
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        #print trainingMat[i,:]
        #print len(trainingMat[i,:])

    testFileList = listdir('testDigits')       
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

handwritingClassTest()
```

### java版本

先看看训练集和测试集：

训练集：

![训练集](https://wizardforcel.gitbooks.io/dm-algo-top10/content/img/20150523213312347.jpg)

测试集：

![测试集](https://wizardforcel.gitbooks.io/dm-algo-top10/content/img/20150523213154774.jpg)


训练集最后一列代表分类（0或者1）

代码实现：

KNN算法主体类：

```java
package Marchinglearning.knn2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * KNN算法主体类
 */
public class KNN {
    /**
     * 设置优先级队列的比较函数，距离越大，优先级越高
     */
    private Comparator<KNNNode> comparator = new Comparator<KNNNode>() {
        public int compare(KNNNode o1, KNNNode o2) {
            if (o1.getDistance() >= o2.getDistance()) {
                return 1;
            } else {
                return 0;
            }
        }
    };
    /**
     * 获取K个不同的随机数
     * @param k 随机数的个数
     * @param max 随机数最大的范围
     * @return 生成的随机数数组
     */
    public List<Integer> getRandKNum(int k, int max) {
        List<Integer> rand = new ArrayList<Integer>(k);
        for (int i = 0; i < k; i++) {
            int temp = (int) (Math.random() * max);
            if (!rand.contains(temp)) {
                rand.add(temp);
            } else {
                i--;
            }
        }
        return rand;
    }
    /**
     * 计算测试元组与训练元组之前的距离
     * @param d1 测试元组
     * @param d2 训练元组
     * @return 距离值
     */
    public double calDistance(List<Double> d1, List<Double> d2) {
        System.out.println("d1:"+d1+",d2"+d2);
        double distance = 0.00;
        for (int i = 0; i < d1.size(); i++) {
            distance += (d1.get(i) - d2.get(i)) * (d1.get(i) - d2.get(i));
        }
        return distance;
    }
    /**
     * 执行KNN算法，获取测试元组的类别
     * @param datas 训练数据集
     * @param testData 测试元组
     * @param k 设定的K值
     * @return 测试元组的类别
     */
    public String knn(List<List<Double>> datas, List<Double> testData, int k) {
        PriorityQueue<KNNNode> pq = new PriorityQueue<KNNNode>(k, comparator);
        List<Integer> randNum = getRandKNum(k, datas.size());
        System.out.println("randNum:"+randNum.toString());
        for (int i = 0; i < k; i++) {
            int index = randNum.get(i);
            List<Double> currData = datas.get(index);
            String c = currData.get(currData.size() - 1).toString();
            System.out.println("currData:"+currData+",c:"+c+",testData"+testData);
            //计算测试元组与训练元组之前的距离
            KNNNode node = new KNNNode(index, calDistance(testData, currData), c);
            pq.add(node);
        }
        for (int i = 0; i < datas.size(); i++) {
            List<Double> t = datas.get(i);
            System.out.println("testData:"+testData);
            System.out.println("t:"+t);
            double distance = calDistance(testData, t);
            System.out.println("distance:"+distance);
            KNNNode top = pq.peek();
            if (top.getDistance() > distance) {
                pq.remove();
                pq.add(new KNNNode(i, distance, t.get(t.size() - 1).toString()));
            }
        }

        return getMostClass(pq);
    }
    /**
     * 获取所得到的k个最近邻元组的多数类
     * @param pq 存储k个最近近邻元组的优先级队列
     * @return 多数类的名称
     */
    private String getMostClass(PriorityQueue<KNNNode> pq) {
        Map<String, Integer> classCount = new HashMap<String, Integer>();
        for (int i = 0; i < pq.size(); i++) {
            KNNNode node = pq.remove();
            String c = node.getC();
            if (classCount.containsKey(c)) {
                classCount.put(c, classCount.get(c) + 1);
            } else {
                classCount.put(c, 1);
            }
        }
        int maxIndex = -1;
        int maxCount = 0;
        Object[] classes = classCount.keySet().toArray();
        for (int i = 0; i < classes.length; i++) {
            if (classCount.get(classes[i]) > maxCount) {
                maxIndex = i;
                maxCount = classCount.get(classes[i]);
            }
        }
        return classes[maxIndex].toString();
    }
}

KNN结点类，用来存储最近邻的k个元组相关的信息

package Marchinglearning.knn2;
/**
 * KNN结点类，用来存储最近邻的k个元组相关的信息
 */
public class KNNNode {
    private int index; // 元组标号
    private double distance; // 与测试元组的距离
    private String c; // 所属类别
    public KNNNode(int index, double distance, String c) {
        super();
        this.index = index;
        this.distance = distance;
        this.c = c;
    }

    public int getIndex() {
        return index;
    }
    public void setIndex(int index) {
        this.index = index;
    }
    public double getDistance() {
        return distance;
    }
    public void setDistance(double distance) {
        this.distance = distance;
    }
    public String getC() {
        return c;
    }
    public void setC(String c) {
        this.c = c;
    }
}
```

KNN算法测试类

```java
package Marchinglearning.knn2;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
/**
 * KNN算法测试类
 */
public class TestKNN {

    /**
     * 从数据文件中读取数据
     * @param datas 存储数据的集合对象
     * @param path 数据文件的路径
     */
    public void read(List<List<Double>> datas, String path){
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(path)));
            String data = br.readLine();
            List<Double> l = null;
            while (data != null) {
                String t[] = data.split(" ");
                l = new ArrayList<Double>();
                for (int i = 0; i < t.length; i++) {
                    l.add(Double.parseDouble(t[i]));
                }
                datas.add(l);
                data = br.readLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 程序执行入口
     * @param args
     */
    public static void main(String[] args) {
        TestKNN t = new TestKNN();
        String datafile = new File("").getAbsolutePath() + File.separator +"knndata2"+File.separator + "datafile.data";
        String testfile = new File("").getAbsolutePath() + File.separator +"knndata2"+File.separator +"testfile.data";
        System.out.println("datafile:"+datafile);
        System.out.println("testfile:"+testfile);
        try {
            List<List<Double>> datas = new ArrayList<List<Double>>();
            List<List<Double>> testDatas = new ArrayList<List<Double>>();
            t.read(datas, datafile);
            t.read(testDatas, testfile);
            KNN knn = new KNN();
            for (int i = 0; i < testDatas.size(); i++) {
                List<Double> test = testDatas.get(i);
                System.out.print("测试元组: ");
                for (int j = 0; j < test.size(); j++) {
                    System.out.print(test.get(j) + " ");
                }
                System.out.print("类别为: ");
                System.out.println(Math.round(Float.parseFloat((knn.knn(datas, test, 3)))));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## 参见

[最近邻居法](https://zh.wikipedia.org/wiki/%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E6%B3%95)

[K Nearest Neighbor 算法](http://coolshell.cn/articles/8052.html)

[数据挖掘十大经典算法(8) kNN: k-nearest neighbor classification ](http://blog.csdn.net/aladdina/article/details/4141127)
