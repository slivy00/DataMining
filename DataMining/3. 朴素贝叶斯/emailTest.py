"""
@file:emailTest.py
@author:姚水林
@time:2018-12-11 21:36:47
@function:
"""
import bayes
import random
from numpy import  *

def spamTest():
    """
    将文件夹spam和ham中分别的25篇右键导入解析为词列表，再构建一个测试集与训练集,
    50篇中再随机选10篇作为测试集，其余20篇作为测试集（留存交叉验证）
    :return: 
    """
    docList = [];classList = [];fullText = []
    for i in range(1,26):
        wordList = bayes.textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = bayes.textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainingSet = list(range(50));testSet=[]
    for i in range(10):#随机选出10篇
        randIndex = int(random.uniform(0,len(trainingSet)))#random.uniform(x, y) 方法将随机生成一个实数，它在 [x,y] 范围内。
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:#遍历训练集中所有的文档
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))#构建词向量
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = bayes.trainNB0(array(trainMat),array(trainClasses))#计算分类所需的概率
    errorCount = 0
    for docIndex in testSet:#遍历测试集
        wordVector = bayes.setOfWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is :',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

if __name__ == '__main__':
    vocabList, p0V, p1V = spamTest()
    print("vocabList=",vocabList,"\np0V=",p0V,"\np1V=",p1V)









