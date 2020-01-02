"""
@file:test.py
@author:Slivy
@time:2019-03-02 15:36:20
@function:
"""
from kNN import *
from numpy import *
import random

# 加载数据集
def loadDataSet(fileName):
    # 数据矩阵
    # dataMat = []
    # 标签向量
    # labelMat = []
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        # strip()表示删除空白符，split()表示分割
        lineArr = line.strip().split(',')
        # dataMat.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2]),float(lineArr[3])])
        # labelMat.append(float(lineArr[4]))
        data.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),float(lineArr[4])])
    # return dataMat, labelMat
    return data

# group, labels = loadDataSet("./iris.csv")
# print("group=",group)
data = loadDataSet("./iris.csv")
print("data=",data)
#从总数据中划分数据为训练数据和测试数据
trainingNum = 120
training = []
test = []
training = random.sample(data, trainingNum)
test = random.sample(data, array(data).shape[0]-trainingNum)
print("training=",training,"\ntest=",test)
trainingData = []
trainingLabels = []
testData = []
testLabels = []
for trainx in training:
    trainingData.append([float(trainx[0]), float(trainx[1]), float(trainx[2]), float(trainx[3])])
    trainingLabels.append([float(trainx[4])])
for testx in test:
    testData.append([float(testx[0]), float(testx[1]), float(testx[2]), float(testx[3])])
    testLabels.append([float(testx[4])])
print("trainingData=",trainingData,"\ntrainingLabels=",trainingLabels,"\ntestData=",testData,"\ntestLabels=",testLabels)
#
correctCountIndex = []
classifyData = []
n = array(testData).shape[0]
for x in testData:
    classify = classify0(x,trainingData,trainingLabels,5)
    classifyData.append(classify)
    # print(testData.index(x))
    if classify == testLabels[testData.index(x)]:
        correctCountIndex.append(testData.index(x))

print("n（测试样本个数）=",n)
print("classifyData（knn分类数据类标）=",classifyData)
print("correctRate(正确率)=%.3f"%(len(correctCountIndex)/n))
