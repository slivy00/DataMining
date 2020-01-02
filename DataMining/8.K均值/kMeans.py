"""
@file:kMeans.py
@author:姚水林
@time:2018-12-20 21:58:25
@function:
"""
from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt

# 加载数据集
def loadDataSet(fileName):
    # 初始化一个空列表
    dataSet = []
    # 读取文件
    fr = open(fileName)
    # 循环遍历文件所有行
    for line in fr.readlines():
        # 切割每一行的数据
        curLine = line.strip().split('\t')
        # 将数据转换为浮点类型,便于后面的计算
        # fltLine = [float(x) for x in curLine]
        # 将数据追加到dataMat
        fltLine = list(map(float,curLine))    # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)
    return dataSet

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
    return centroids, clusterAssment# 返回所有的类质心与点分配结果

def biKmeans(dataMat, k, distMeas=distEclud):
    # 获取样本数和特征数
    m, n = shape(dataMat)
    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m, 2)))
    # 计算整个数据集的质心,并使用一个列表来保留所有的质心
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList = [centroid0]
    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataMat[j, :]) ** 2
    # 对簇不停的进行划分,直到得到想要的簇数目为止
    while (len(centList) < k):
        # 初始化最小SSE为无穷大,用于比较划分前后的SSE
        lowestSSE = inf
        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(centList)):
            # 对每一个簇,将该簇中的所有点堪称一个小的数据集
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
            # kMeans会生成两个质心(簇),同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 将误差值与剩余数据集的误差之和作为本次划分的误差
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            # 如果本次划分的SSE值最小,则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果
        # 调用kmeans函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 更新为最佳质心
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加bestNewCents的第二个质心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配最好簇下的数据(质心)以及SSE
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

if __name__ == '__main__':
    dataMat = mat(loadDataSet('./testSet.txt'))
    print("dataMat=",dataMat)
    # min1 = min(dataMat[:,0])
    # min2 = min(dataMat[:,1])
    # max1 = max(dataMat[:,0])
    # max2 = max(dataMat[:,1])
    # print("min1=%f,min2=%f"%(min1,min2))
    # print("max1=%f,max2=%f" %(max1 , max2))
    # centroids = randCent(dataMat,2)
    # print("centroids=",centroids)
    # distAB = distEclud(dataMat[0],dataMat[1])
    # print("distAB=",distAB)
    myCentroids,clustAssing = kMeans(dataMat,4)
    print("myCentroids=",myCentroids)
    print("clustAssing=",clustAssing)
    # dataMat2 = mat(loadDataSet('testSet2.txt'))
    # centList,myNewClustAssing = biKmeans(dataMat2,3)
    # print("centList=",centList)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(centList[:1],'o')
    # plt.show()



