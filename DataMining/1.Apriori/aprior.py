"""
@file:apriori.py
@author:姚水林
@time:2018-12-09 15:53:14
@function:
1、产生数据集
2、构建所有候选项集单元素的并集
3、取得满足minsup的集合项和算出Ck各项支持度的字典
4、获得候选项集Ck
5、生成频繁项集列表Lk(C1->L1（scanD）->Ck（aprioriGen）->L(scanD))
6、生成关联规则（计算可信度；生成候选规则集合）
"""

def loadDataSet():
    """
    产生训练数据集
    :return: 数据集(list)
    """
    return [
        [1,3,4],
        [2,3,5],
        [1,2,3,5],
        [2,5]
    ]

def createC1(dataSet):
    """
    构建所有候选项集单元素的并集
    :param dataSet: 
    :return: 返回大小为1的所有候选项集的frozenset集合（不可改，后需要将这些集合作为字典键使用，set无法实现）
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))#map会根据提供的函数对指定序列做映射。

def scanD(D,Ck,minSupport):
    """
    取得满足minsup的集合项和算出Ck各项支持度的字典
    :param D: 数据集dataSet
    :param Ck: 候选集合C1...
    :param minSupport: 最小支持度
    :return: 满足minsup的集合Lk和包含了C(k-1)中各项支持度的字典
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):#issubset:判断can集合是否在tid集合中
                if can not in ssCnt:ssCnt[can] = 1 #has_key判断键是否存在于字典中
                else : ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    #print(ssCnt)
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

def aprioriGen(Lk,k):
    """
    获得候选项集Ck
    :param Lk: 频繁项集列表Lk
    :param k: 生成候选项集元素个数k
    :return: 候选项集Ck
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk): #range:创建一个整数列表，不包括结尾值
            L1 = list(Lk[i])[:k-2];L2 = list(Lk[j])[:k-2] #{{5,2},{5,3}...}判断集合中第一个是否相同来合并后面的，减少项
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    """
    生成频繁项集列表Lk(C1->L1（scanD）->Ck（aprioriGen）->L(scanD))
    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return: 频繁项集列表L和包含了C(k-1)中各项支持度的字典
    """
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    #print("L=",L)
    k = 2
    while (len(L[k-2]) > 0):#L最后一项为空
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK) # 更新字典中的键/值对
        L.append(Lk)
        k += 1
    return L,supportData

def  generateRules(L,supportData,minConf=0.7):
    """
    生成关联规则
    :param L: 频繁项集
    :param supportData: 包含频繁项集支持度数据的字典
    :param minConf: 最小可信度
    :return: 包含对应关联规则和conf的列表数据
    """
    bigRuleList = []
    for i in range(1,len(L)):#i=1从第二组开始，因为第一组每项只有一个元素无法构成关联规则
        for freqSet in L[i]:
            #print("freqSet=",freqSet)
            H1 = [frozenset([item]) for item in freqSet]
            # print("H1=",H1)
            if (i > 1):#两个以上的元素组成的集合需要生成候选规则
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcConf(freqSet,H,supportData,br1,minConf = 0.7):#frozenset({2, 3}),supportData,[frozenset({2}), frozenset({3})],bigRuleList
    """
    计算可信度
    :param freqSet: 包含多个元素的集合
    :param H: 频繁项集中的元素列表
    :param supportData: 包含频繁项集支持度数据的字典
    :param br1: 关联规则列表的数组
    :param minConf: 最小可信度
    :return: 对应关联规则和conf的列表数据
    """
    prunedH = []
    for conseq in H:
        conf  = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7):#frozenset({2,3,5}),H1= [frozenset({2}), frozenset({3}),frozenset({5})]
    """
    生成候选规则集合
    :param freqSet: 包含多个元素的集合
    :param H:  频繁项集中的元素列表
    :param supportData: 包含频繁项集支持度数据的字典
    :param br1: 关联规则列表的数组
    :param minConf: 最小可信度
    :return: none
    """
    m = len(H[0])
    if(len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H,m+1)
        # print("Hmp1=",Hmp1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if(len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)


if __name__ == '__main__':
    dataSet = loadDataSet()
    print("dataSet=",dataSet)
    C1 = createC1(dataSet)
    print("C1=",C1)
    D = list(map(set,dataSet))
    L1,suppData0 = scanD(D,C1,0.5)
    print("L1=",L1)
    print("suppData0=",suppData0)
    C2 = aprioriGen(L1,2)
    print("C2=",C2)
    L ,suppData =apriori(dataSet,0.5)
    print("L=",L)
    print("suppData=",suppData)
    rules = generateRules(L,suppData,minConf=0.5)
    print("rules=",rules)
