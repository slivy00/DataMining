"""
@file:regression.py
@author:姚水林
@time:2018-12-16 11:04:50
@function:
"""
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    打开一个用tab键分隔的文本文件,默认文件格式:X0,X1,Y
    :param fileName: 文件名
    :return: X列表dataMat，Y列表labelMat
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1 # 获得所有标签
    dataMat = []; labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat

def standRegres(xArr,yArr):
    """
    得到X数据集对应的系数最优解列表
    :param xArr: X列表
    :param yArr: Y列表
    :return: X0,X1...系数的最优解,ws[0] ws[1].... 
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat  #计算XT*X
    if linalg.det(xTx) == 0.0:  # 判断行列式是否为零，为零则计算逆矩阵出现错误
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)  #(XT*X)^(-1) * XTy
    return ws

def plotRegression(XArr,YArr):
    """
    画出原始数据和拟合曲线的效果图
    :param XArr: X列表
    :param YArr: Y列表
    :return: none
    """
    xMat = mat(XArr)
    yMat = mat(YArr)
    yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])#绘制散点图
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

def lwlr(testPoint,xArr,yArr,k=1.0):
    """
    给定x空间中的一点，计算出对应的预测值yHat
    :param testPoint: 
    :param xArr: X数据集
    :param yArr: Y数据集
    :param k: 控制权重衰减的速度
    :return: 点testPoint对应的预测值
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建对角权重矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]     # 权重值大小以指数级衰减
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    """
    给定X数据集一点，计算出对应的预测值yHat
    :param testArr: X数据集
    :param xArr: X数据集
    :param yArr: Y数据集
    :param k: 控制权重衰减的速度
    :return: X数据集预测值yHat列表
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def plotlwlrRegression(xArr,yArr,yHat):
    """
    绘制局部加权线性回归结果
    :param xArr: X数据集
    :param yArr: Y数据集
    :param yHat: 对应XArr中点的预测值
    :return: none
    """
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')  # 绘制散点图
    plt.show()

def rssError(yArr,yHatArr):
    """
    使用平方误差，分析预测误差的大小
    :param yArr: y数据集（真实值）
    :param yHatArr: 预测值
    :return: 平方误差
    """
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    """岭回归
    计算岭回归系数
    :param xMat: X矩阵数据集
    :param yMat: Y矩阵数据集
    :param lam: λ值
    :return: 岭回归系数
    """
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    """
    用于在一组λ上做测试
    :param xArr: x数组
    :param yArr: y数组
    :return: 
    """
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)#数据的标准化过程，具体过程就是所有特征都减去各自的均值并处理方差
    yMat = yMat - yMean  # 要消去X0，取Y的平均值。
    xMeans = mean(xMat,0)  # calc的意思是减去它。
    xVar = var(xMat,0)  # 然后再除以Xi的calc方差。
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    print("dataMat=",xArr)
    print("labelMat=",yArr)
    ws = standRegres(xArr,yArr)
    print("ws=",ws)
    plotRegression(xArr,yArr)
    yHat = lwlrTest(xArr,xArr,yArr,0.003)
    # plotlwlrRegression(xArr,yArr, yHat)




















