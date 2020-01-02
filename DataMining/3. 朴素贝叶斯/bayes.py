"""
@file:bayes.py
@author:姚水林
@time:2018-12-10 16:39:48
@function:
1、创建实验数据loadDataSet
2、创建数据中词的并集列表createVocabList
3、词集模型setOfWords2Vec
4、词袋模型bagOfWords2VecMN
5、文档矩阵wordsAll
6、朴素贝叶斯分类器训练函数trainNB0
7、朴素贝叶斯分类函数classifyNB
8、切割字符串函数textParse
"""
from numpy import *
import  re
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
        else: print("the word:%s is not in my Vocabulary!"%(word))
    return returnVec

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
    :return: 为0的概率向量集p0Vect，为1的概率向量集p1Vect，类别标签构成的向量集中为1的概率
    """
    numTrainDocs = len(trainMatrix)#数据集并集向量项数
    numWords = len(trainMatrix[0])#数据集并集向量第一项中元素个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = zeros(numWords);p1Num = zeros(numWords)#初始化个数[0.,0.,...]
    # p0Denom = 0.0;p1Denom = 0.0 #初始化0和1的分母
    p0Num = ones(numWords);p1Num = ones(numWords)#避免一个概率为0，最后值也为0了,p(W0|1)P(W1|1)...
    p0Denom = 2.0;p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    p1Vect = log(p1Num / p1Denom)#避免避免下溢出，当太多很小的数相乘最后四舍五入时会得到0,采用乘积取对数的方式避免
    p0Vect = log(p0Num / p0Denom)#f(x)和ln(f(x))
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    """
    朴素贝叶斯分类器函数.使用NumPy的数组来计算两个向量相乘的结果,这里的相乘是指对应元素相乘，即先将两个向量中的第1个元素相乘，然后将第2个元素相乘，以此类推。接下来将词汇表
    中所有词的对应值相加，然后将该值加到类别的对数概率上。最后，比较类别的概率返回大概率对应的类别标签
    :param vec2Classify: 要分类的向量
    :param p0Vec: 为0的概率向量集p0Vect
    :param p1Vec: 为1的概率向量集p1Vect
    :param pClass1: 类别标签构成的向量集中为1的概率
    :return: 类别标签0/1
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else :
        return 0


def testingNB():
    """
    测试NB
    :return: none
    """
    listOPosts, listClasses = loadDataSet()
    print("listOPosts=",listOPosts,"\nlistClasses=",listClasses)
    myVocabList = createVocabList(listOPosts)
    print("myVocabList=",myVocabList,"\nlen(myVocabList)=",len(myVocabList))
    trainMat1 = bagOfWords2VecMN(myVocabList,listOPosts[0])
    print("trainMat1=",trainMat1,"\nlen(trainMat1)=",len(trainMat1))
    trainMat = wordsAll(myVocabList, listOPosts)
    print("trainMat=",trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # testEntry = ['love','my','dalmation']
    # thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    # print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    # testEntry = ['stupid', 'garbage']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    """
    得到分割后的字符列表
    :param bigString: 需要分割的字符串(需要传入的是单引号分割的字符串)
    :return: 全部是小写字符的列表
    """
    listOfTokens = re.split(r'\W+',bigString)
    # listOfTokens = bigString.split()
    return  [tok.lower() for tok in listOfTokens if len(tok) > 2]

if __name__ == '__main__':
    # listOPosts,listClasses = loadDataSet()
    # print("listOPosts=", listOPosts)
    # print("listClasses=", listClasses)
    # myVocabList = createVocabList(listOPosts)
    # print("myVocabList=",myVocabList)
    # returnVec = setOfWords2Vec(myVocabList,listOPosts[0])
    # print("returnVec=",returnVec)
    # trainMat = wordsAll(myVocabList,listOPosts)
    # print("trainMat",trainMat)
    # p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    # print("pAb=",pAb);print("p0V=",p0V);print("p1V=",p1V)
    testingNB()
    # splitString = textParse('This is test string  maybe or not I can!')
    # print(splitString)
    # testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # p0,p1 = classifyNBP1(thisDoc,p0V,p1V,pAb)
    # print("p0=",p0,"\np1=",p1)



