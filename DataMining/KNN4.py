
from numpy import *
import operator
import numpy as np
import random as rand
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
#knn
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
#kMeans
def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离
    :param vecA: 
    :param vecB: 
    :return: 欧式距离值
    """
    return sqrt(sum(power(vecA - vecB, 2)))
def randCent(dataMat, k):
    """
    :param dataMat: 
    :param k: 簇数目
    :return: 质心
    """
    # 获取样本数与特征值
    m, n = shape(dataMat)
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = mat(zeros((k, n)))
    # 循环遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataMat[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # 计算每一列的质心,并将值赋给centroids
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids
def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    """
    :param dataMat: 
    :param k: 簇数目
    :param distMeas: 
    :param createCent: 
    :return: 中心点centroids，点分配结果clusterAssment。
    """
    # 获取样本数和特征数
    m, n = shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = mat(zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment  # 返回所有的类质心与点分配结果

def selectPart(dataSet,num):
    """
    :param dataSet: 数据
    :param num: 随机去掉的个数
    :return: 返回选取的个数列表
    """
    dataSetSize = array(dataSet).shape[0]
    ran_li = rand.sample(dataSet, dataSetSize - num)
    # print(array(ran_li).shape[0])
    # # print("dataSetSize=",dataSetSize)
    # diffMat = tile(cent, (dataSetSize, 1)) - dataSet
    # sqDiffMat = diffMat ** 2
    # sqDistances = sqDiffMat.sum(axis=1)  # 按照行的方向相加
    # distances = sqDistances ** 0.5
    # sortedDistIndicies = distances.argsort()  # 按下标排序
    # # print("sortedDistIndicies=",sortedDistIndicies)
    # # print(sortedDistIndicies[:num])
    # distancePart = []
    # for index in sortedDistIndicies[:dataSetSize-num]:
    #     distancePart.append([float(dataSet[index][0]), float(dataSet[index][1]), float(dataSet[index][2]), float(dataSet[index][3])])
    return ran_li

data = loadDataSet("./iris.csv")
np.random.shuffle(data)
#从总数据中划分数据为训练数据和测试数据
trainingNum = 120
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
#将训练数据按类标分三类，每类取离质心最近的一部分点
train_0 = []
trainLabels_0 = []
train_1 = []
trainLabels_1 = []
train_2 = []
trainLabels_2 = []
for train in training:
    if train[4] == 0:
        train_0.append([float(train[0]), float(train[1]), float(train[2]), float(train[3])])
        trainLabels_0.append(float(train[4]))
    if train[4] == 1:
        train_1.append([float(train[0]), float(train[1]), float(train[2]), float(train[3])])
        trainLabels_1.append(float(train[4]))
    if train[4] == 2:
        train_2.append([float(train[0]), float(train[1]), float(train[2]), float(train[3])])
        trainLabels_2.append(float(train[4]))

print("train_0=",train_0,"\ntrainLabels_0=",trainLabels_0,"\ntrain_1","\ntrainLabels_1=",trainLabels_1,train_1,"\ntrain_2",train_2,"\ntrainLabels_2=",trainLabels_2)

# print(cent0.tolist()[0])
#随机去掉点数
distancePartNum = 10
distancePart0 = selectPart(train_0,distancePartNum)
distancePart1 = selectPart(train_1,distancePartNum)
distancePart2 = selectPart(train_2,distancePartNum)
cent0=randCent(mat(distancePart0),1)
cent1=randCent(mat(distancePart1),1)
cent2=randCent(mat(distancePart2),1)
print("cent0=",cent0,"\ncent1=",cent1,"\ncent2=",cent2 )
# print("distancePart0=",distancePart0,"\ndistancePart1=",distancePart1,"\ndistancePart2=",distancePart2)
centTrainTotal = []
centLabels = []
centLabels.extend([0]+[1]+[2])
centTrainTotal.extend(cent0.tolist()+cent1.tolist()+cent2.tolist())
# print("centTrainTotal=",centTrainTotal,"\ncentLabels=",centLabels)
# print("distancePartTotal=",distancePartTrainTotal,"\ndistancePartLabels=",distancePartLabels)
# #
correctCountIndex = []
classifyData = []
for x in testData:
    classify = classify0(x,centTrainTotal,centLabels,1)
    classifyData.append(classify)
    if classify == testLabels[testData.index(x)]:
        correctCountIndex.append(testData.index(x))

print("trainingNum（每类除去局外点后选取质心个数）=",array(centTrainTotal ).shape[0])
print("testNum（随机选取测试样本个数）=",array(testData).shape[0])
print("classifyData（knn分类数据类标）=",classifyData)
print("correctRate(正确率)=%.3f"%(len(correctCountIndex)/array(testData).shape[0]))
end = time.clock()
print("time=",end - start)