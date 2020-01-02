"""
 @file : example.py 
 @author : 姚水林
 @time : 2019/5/2 16:28
 @function:
"""
from numpy import *
import math
import csv
import codecs
import pandas as pd

def loadDataSet():
    """
    创建一些实验样本
    :return: 变量进行了词条切分后的文档列表postingList和类别标签的列表classVec
    """
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1] # 1代表侮辱性文字，0代表正常言论
    return postingList,classVec
def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 数据集
    :return: 返回一个列表包含文档中全部不重复的词list(vocabSet)
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
def bagOfWords2VecMN(vocabList,inputSet):
    """词袋模型
    若inputSet数据值在全部数据集的并集中，则将对应位置+1
    :param vocabList: 数据集并集
    :param inputSet: 原始数据集列表的一项
    :return: 文档向量returnVec
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # 将存在值+1
    return returnVec
def wordsAll(vocabList,listOPosts):
    """
    得到原始数据集列表对应的向量集,向量集的每一项对应全部数据并集的向量集（位数相等）
    :param vocabList: 数据集并集
    :param listOPosts: 原始数据集
    :return: 文档矩阵
    """
    trainMat = []
    for postinDoc in  listOPosts:
        trainMat.append(bagOfWords2VecMN(vocabList,postinDoc))
    return trainMat
def trainNB0(trainMatrix,trainCategory):
    """
    得到每个数据集并集各项的为0的概率向量集p0Vect，为1的概率向量集p1Vect，类别标签构成的向量集中为1的概率
    :param trainMatrix: 文档矩阵
    :param trainCategory: 类别标签构成的向量集
    :return: 0类下每个词出现的概率向量集p0Vect，1类下每个词出现的概率向量集p1Vect，类别标签构成的向量集中为1的概率
    """
    numTrainDocs = len(trainMatrix)#数据集并集向量项数
    # print("numTrainDocs=",numTrainDocs) #6
    numWords = len(trainMatrix[0])#数据集并集向量第一项中元素个数
    # print("numWords=",numWords) #32
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = zeros(numWords);p1Num = zeros(numWords)# # 构造单词出现次数列表,初始化个数[0.,0.,...]
    # print("p0Num=",p0Num,"\np1Num=",p1Num) #32个0
    # p0Denom = 0.0;p1Denom = 0.0;pwDenom = 0.0 #整个数据集单词出现总数,初始化0和1的分母
    p0Num = ones(numWords);p1Num = ones(numWords)#避免一个概率为0，最后值也为0了,p(W0|1)P(W1|1)...
    pwNum = ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0;pwDenom = 2.0
    for i in range(numTrainDocs):
        # pwNum += trainMatrix[i]
        # pwDenom += sum(trainMatrix[i])
        # print("pwDenom=",pwDenom)
        # print("pwNum=",pwNum)
        if trainCategory[i] == 1:# 是否是侮辱性文件
            p1Num += trainMatrix[i]# 如果是侮辱性文件，对侮辱性文件的向量进行加和
            # print("p1Num=",p1Num)
            p1Denom += sum(trainMatrix[i])# 对向量中的所有元素进行求和，也就是计算所有侮辱性文件中出现的单词总数
            # print("p1Denom=",p1Denom)#19
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            # print("p0Denom=", p0Denom)#24
    # 类别1，即侮辱性文档的[p(w0 | c1)p(w1 | c1)p(w2 | c1)...p(wn | c1)] 列表
    # 即 在1类别下，每个单词出现的概率
    # 类别0，即正常文档的[p(w0 | c0)p(w1 | c0)p(w2 | c0)...p(wn | c0).]列表
    # 即 在0类别下，每个单词出现的概率
    # p0Vect = p0Num / p0Denom
    # p1Vect = p1Num / p1Denom
    # pwVect = pwNum / pwDenom
    p1Vect = log(p1Num / p1Denom)#避免下溢出，当太多很小的数相乘最后四舍五入时会得到0,采用乘积取对数的方式避免
    p0Vect = log(p0Num / p0Denom)#f(x)和ln(f(x))
    # print("pwVect=",pwVect)
    return p0Vect,p1Vect,pAbusive
def setOfWords2Vec(vocabList,inputSet):
    """词集模型
    若inputSet数据值在全部数据集的并集中，则将对应位置置1
    :param vocabList: 数据集并集
    :param inputSet: 原始数据集列表的一项
    :return: 文档向量returnVec（1或0）(inputSet若在vocabList中则置为1，不在为0)
    """
    returnVec = [0] * len(vocabList)
    # print(returnVec)
    for word in inputSet:
        if word in vocabList:
            # print(vocabList.index(word))
            returnVec[vocabList.index(word)] = 1 # 将存在值置1
        # else: print("the word:%s is not in my Vocabulary!"%(word))
    # print("returnVec=",returnVec)
    return returnVec
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # print("vec2Classify * p1Vec=\n",vec2Classify * p1Vec,"\nvec2Classify * p0Vec=\n",vec2Classify * p0Vec)
    # print("p1=",math.exp(sum(p1Vec) + log(pClass1) - sum(pwVect)),"\np0=",math.exp(sum(p0Vec) + log(1.0 - pClass1) - sum(pwVect)))
    # print(sum(log(p1Vec)))
    # p1 = math.exp(sum(vec2Classify * log(p1Vec)))*(pClass1)/sum(pwVect)
    # p1 = sum(vec2Classify * p1Vec) + log(pClass1) - sum(pwVect)
    p1 = math.exp(sum(vec2Classify * p1Vec) + log(pClass1))
    p0 = math.exp(sum(vec2Classify * p0Vec) + log(1.0 - pClass1))
    P = p1 /(p0 + p1)
    # print(sum(vec2Classify * p0Vec) + log(1.0 - pClass1))
    return P
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    # name = [ 'Pred']
    # test = pd.DataFrame(columns=name, data=datas)
    # test = test.reindex(index=list(range(1, test.shape[0])))
    # test.to_csv(file_name, encoding='gbk')
    # print("保存文件成功，处理结束")
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in datas:
            writer.writerow(row)
    print("保存文件成功，处理结束")

if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    print("postingList=",postingList,"\nclassVec=",classVec)
    myVocabList = createVocabList(postingList)
    print("myVocabList=",myVocabList)
    trainMat1 = bagOfWords2VecMN(myVocabList,postingList[0])
    print("trainMat1=",trainMat1)
    trainMat = wordsAll(myVocabList, postingList)
    print("trainMat=", trainMat)
    p0V, p1V,pAb = trainNB0(trainMat, classVec)
    # print("p0V=",p0V,"\np1V=",p1V,"\npAb=",pAb)
    # testEntry = ['love', 'my', 'dalmation']
    # testEntry = ['stupid', 'garbage']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # classify = classifyNB(thisDoc,p0V,p1V,pAb)
    # print("classify=",classify)
    testEntry = [
        ['love', 'my', 'dalmation'],
        ['big', 'help', 'dalmation'],
        ['stupid', 'garbage']
    ]
    indexId =[1,2,3]
    classifyMat = []
    for i,x in enumerate(testEntry):
        thisDoc = array(setOfWords2Vec(myVocabList, x))
        classify = classifyNB(thisDoc, p0V, p1V, pAb)
        # print("classify=",classify)
        classifyMat.append([indexId[i],classify])
    print("classifyMat=",classifyMat)
    data_write_csv("./example.csv",classifyMat)


