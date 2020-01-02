"""
@file:trees.py
@author:姚水林
@time:2018-12-08 21:05:15
@function:ID3算法
1、产生训练数据集
2、计算香农熵
3、划分数据集
4、选择最好的信息增益
5、递归构造决策树
6、存读决策树
7、取得次数最多的类标名
"""
from math import log
import operator
import pickle
import treePlotter

def createDataSet():
    """
    产生训练数据集
    :return: 数据集（list）
    """
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['first','second']
    return dataSet,labels

def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet: 给定数据集
    :return: 香农熵值（float）
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] #默认数据最后一项是类标
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # print("labelCounts=",labelCounts)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    """
    划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征（根据列表中的那一项来划分）
    :param value: 特征划分值
    :return: 划分后的数据，不包括特征划分那一项（list）
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value: #抽取数据
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])# extend和append方法。这两个方法功能类似，但是在处理多个列表时，这两个方法的处理结果是完全不同的。
            retDataSet.append(reducedFeatVec)   #append 添加一个列表直接将列表添加到末尾作为一个元素，extend则添加有多个元素
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet: 数据集（要求list，要求每个实例的类标）
    :return: 最好的划分方式的下标(int)
    """
    numFeatures = len(dataSet[0]) - 1 #除去类标不作为特征值 numFeatures=[0,1]
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        #print("featList=",featList)
        uniqueVals = set(featList) #转变为集合并去掉重复项
        #print("uniqueVals=",uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals: #计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            #print("infoGain=",infoGain)
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    返回次数最多的类标名
    :param classList: 类标列表（list）
    :return: 分类名(string)
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
        #print("classCount=",classCount)
        #.items(),对字典操作,返回可遍历的(键, 值) 元组数组。
        #key用列表元素的某个属性或函数进行作为关键字来排序
        #reverse=True:降序,默认=False
        #operator.itemgetter(1):根据第二个域进行排序,operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    """
    递归构造决策树
    :param dataSet:数据集（要求list，要求每个实例的类标）
    :param labels: 标签列表（list)
    :return: 决策树字典(dict)
    """
    classList = [example[-1] for example in dataSet]
    #递归停止条件:1.类标完全相同则停止划分,2.使用完了所有特征，仍然不能将数据集划分为仅包含唯一类别的分组，挑选出现次数最多的返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        #print(subLabels)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

def storeTree(inputTree,filename):
    """
    使用pickle来存储决策树
    :param inputTree: 存放树自字典
    :param filename: 文件名
    :return: none
    """
    fw = open(filename,'wb')
    # inputTree = inputTree.encode('utf-8')
    pickle.dump(inputTree,fw,0)#obj对象序列化存入已经打开的file,默认0
    fw.close()

def grabTree(filename):
    """
    取得存放树
    :param filename: 文件名
    :return: 序列化数据
    """
    fr =  open(filename,'rb')
    return pickle.load(fr)#将file中的对象序列化读出。

def classify(inputTree,featLabels,testVec):
    """
    测试函数
    :param inputTree: 生成好的决策树
    :param featLabels: 类属性列表（list）
    :param testVec: 测试数据列表（list）
    :return: 类标
    """
    firstStr = list(inputTree.keys())[0] #no surfacing
    secondDict = inputTree[firstStr] # {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    featIndex = featLabels.index(firstStr) # 0 1  将标签字符串转化为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict': #type().__name__ == 'dict' 判断该类型是否为dict
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel

if  __name__ == '__main__':
    myDat,labels = createDataSet()
    print("myDat=",myDat)
    shannonEnt = calcShannonEnt(myDat)
    print("shannonEnt=",shannonEnt)
    data = splitDataSet(myDat,0,1)
    print("data=",data)
    data2 = splitDataSet(myDat, 0, 0)
    print("data2=", data2)
    bestFeatureToSplit = chooseBestFeatureToSplit(myDat)
    print("bestFeatureToSplit=",bestFeatureToSplit)
    classList = [example[-1] for example in myDat]
    print("classList=",classList)
    print("sortedClassCount=",majorityCnt(classList))
    myTree = createTree(myDat, labels)
    print("myTree=",myTree)
    storeTree(myTree,'classifierStorage.txt')
    print("grabTree=",grabTree('classifierStorage.txt'))
    treePlotter.createPlot(myTree)
    #测试数据
    labels2 = ['first', 'second']
    print("test=[1,0]:",classify(myTree,labels2,[1,0]))





