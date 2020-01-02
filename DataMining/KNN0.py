
from numpy import *
import operator
import numpy as np
import time

start = time.clock()
# 加载数据集
def loadDataSet(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        # strip()表示删除空白符，split()表示分割
        lineArr = line.strip().split(',')
        fltLine = list(map(float, lineArr))  # 映射所有的元素为 float（浮点数）类型
        data.append(fltLine)
    return data

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


data = loadDataSet("./iris.csv")
np.random.shuffle(data)
#从总数据中划分数据为训练数据和测试数据
trainingNum = 4000
training = []
test = []
training = data[:trainingNum]
test = data[trainingNum:]
print("training=",training,"\ntest=",test)
trainingData = []
trainingLabels = []
testData = []
testLabels = []
for trainx in training:
    trainingData.append([float(trainx[0]), float(trainx[1]), float(trainx[2]), float(trainx[3])])
    trainingLabels.append(float(trainx[4]))
for testx in test:
    testData.append([float(testx[0]), float(testx[1]), float(testx[2]), float(testx[3])])
    testLabels.append(float(testx[4]))
print("trainingData=",trainingData,"\ntrainingLabels=",trainingLabels,"\ntestData=",testData,"\ntestLabels=",testLabels)
correctCountIndex = []
classifyData = []
for x in testData:
    classify = classify0(x,trainingData,trainingLabels,3)
    classifyData.append(classify)
    if classify == testLabels[testData.index(x)]:
        correctCountIndex.append(testData.index(x))

print("trainingNum（随机选取训练样本个数）=",array(trainingData).shape[0])
print("testNum（随机选取测试样本个数）=",array(testData).shape[0])
print("classifyData（knn分类数据类标）=",classifyData)
print("correctRate(正确率)=%.3f"%(len(correctCountIndex)/array(testData).shape[0]))
end = time.clock()
print("time=",end - start)