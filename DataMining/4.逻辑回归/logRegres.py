"""
@file:logRegres.py
@author:姚水林
@time:2018-12-13 15:32:46
@function:
"""
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    """
    载入数据，格式：每行前两个值：X1，X2，第三个值为数据对应的类别标签,X0的初始值设置为了1.0
    :return: X0=1.0,X1=第一个值，X2=第二个值的列表dataMat,第三个值组成的标签列表labelMat
    """
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # print(lineArr)
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    """
    sigmoid函数
    :param inX: 参数x值计算sigmoid函数值
    :return: sigmoid函数值
    """
    return 1.0 / (1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    """
    梯度上升算法
    :param dataMatIn: 数据集
    :param classLabels: 标签列表
    :return: 最佳回归参数矩阵
    """
    dataMatrix = mat(dataMatIn) #转换为NumPy数组进行运算
    # print("dataMatrix=",dataMatrix)
    labelMat = mat(classLabels).transpose() #矩阵转置
    # print("labelMat=", labelMat)
    m,n = shape(dataMatrix) #查看矩阵或者数组的维数。
    alpha = 0.001 #目标移动的步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1))
    # print(weights)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(wei):
    """
    分析数据，画出决策边界
    :param wei: 最佳回归参数矩阵
    :return: none
    """
    weights = wei # 将numpy矩阵转换为数组
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if  int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels):
    """
    随机梯度上升算法，与梯度上升算法的区别：第一，后者的变量h和误差error都是向量，而前者则全是数值；
    第二，前者没有矩阵的转换过程，所有变量的数据类型都是NumPy数组。
    :param dataMatrix: 数据集
    :param classLabels: 标签列表
    :return: 最佳回归参数矩阵
    """
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights2 = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights2))
        error = classLabels[i] - h
        weights2 = weights + alpha*error*dataMatrix[i]
    return weights2

def stocGradAscent1(dataMatrix,classLabels,numIter=150):

    m,n = shape(dataMatrix)
    weights2 = ones(n)
    for  j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[i]*weights2))
            error = classLabels[i] - h
            weights2 = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights2

if __name__ == '__main__':
    dataMat,labelMat=loadDataSet()
    print("dataMat=",dataMat)
    print("labelMat=",labelMat)
    weights = gradAscent(dataMat,labelMat)
    print("weights=",weights)
    plotBestFit(weights.getA())
    # dataArr, labelMat2 = loadDataSet()
    # weights2 = stocGradAscent0(array(dataArr),labelMat2)
    # plotBestFit(weights2)










