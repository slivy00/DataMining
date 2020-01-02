"""
@file:kNN.py
@author:姚水林
@time:2018-12-16 19:21:41
@function:
"""
from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels



def classify0(inX, dataSet, labels, k):
    dataSetSize = array(dataSet).shape[0]
    # print("dataSetSize=",dataSetSize)
    # print("tile(inX, (dataSetSize,1))=",tile(inX, (dataSetSize,1)))
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # print("diffMat=",diffMat)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#按照行的方向相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #按下标排序
    classCount={} #字典类型
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # print("classCount=",classCount)
    # print("classCount.items()=",classCount.items())
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print("sortedClassCount=",sortedClassCount)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    group,labels = createDataSet()
    print("group=",group,"\nlabels=",labels)
    # print("group[1]=",group.shape[1])
    classify0 = classify0([0,0.5],group,labels,3)
    print("classify0=",classify0)
    print("__name__=",__name__)

