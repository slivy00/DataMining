'''
1. 遍历数据库，得到所有数据项构成的并集（也就是得到图1的C1层）
2. 计算Ck层中每个元素的支持度（该过程可用Hash表优化），删除不符合的元素，将剩下的元素排序，并加入频繁项集R
3. 根据融合规则将Ck层的元素融合得到Ck+1,
4. 重复2,3步直到某一层元素融合后得到的是空集
5. 遍历R中的元素，设该元素为A={a1，a2......，ak}
6. 生成I1层规则，即{x|x属于A且≠ai} →{ai}
7. 计算该层所有规则的“置信度”，删除不符合的规则，将剩下的规则作为结果输出
8. 生成下一层的规则，计算“置信度”，输出结果。
'''

import numpy as np
import  itertools

#生成原始数据集，用于测试
def loadDataSet():
    return [
        ["a","b","c"],
        ["b","c","d"],
        ["a","b","c","d"],
        ["b","d"]
    ]
#获取整个数据集中的元素并集
def createC1(dataSet):
    C1 = set([])
    for item in dataSet:
        C1 = C1.union(set(item)) #set.union方法返回两个集合的并集
    return [frozenset([i]) for i in C1]#set(可变集合)与frozenset(不可变集合)

#输入数据集（dataSet)和由第k-1层数据融合后得到的第k层数据集Ck
#用最小支持度（minSupport），对Ck进行过滤得到第k层剩下的数据集合Lk
support_dic = {} #用来存储每一个项的支持度
def getLk(dataSet,Ck,minSupport):
    global support_dic #给定义在函数外的变量赋值得先声明该变量是全局的
    Lk = {}
    for item in dataSet:
        for Ci in Ck:
            if Ci.issubset(item):#issubset:判断Ci集合是否在item集合中
                if not Ci in Lk:
                    Lk[Ci] = 1
                else:
                    Lk[Ci] += 1
    Lk_return = [] #用来存储大于minSupport的项
    for Li in Lk: # Li-> frozenset({1})，frozenset({2})，frozenset({3})...
        support_Li = Lk[Li] / float(len(dataSet)) #计算支持度
        if support_Li >= minSupport:
            Lk_return.append(Li)
            support_dic[Li] = support_Li #存储每一项的支持度的值。
    return Lk_return    #返回存储大于minSupport的项
#将经过支持度过滤的第k层数据即可Lk融合
#得到第k+1层原始数据Ck1
def genLk1(Lk):
    Ck1 = []
    for i in range(len(Lk) - 1): # range() 函数可创建一个整数列表，不包括结尾值
        for j in range(i + 1,len(Lk)):
            # sorted() 函数对所有可迭代的对象进行排序操作
            #ist 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
            if sorted(list(Lk[i]))[0:-1] == sorted(list(Lk[j]))[0:-1]:# [0:-1];切第一个元素到倒数第二个元素（包括）
                Ck1.append(Lk[i] | Lk[j]) #将Lk[i]和Lk[j]组合一个集合 #第一轮:
    return Ck1
#遍历所有二阶及以上的频繁项集合
def genItem(freqSet):
    for i in range(1,len(freqSet)):
        for freqItem in freqSet[i]:
            genRule(freqItem)
#输入一个频繁项，根据“置信度”生成规则
#采用了递归，对规则树进行剪枝
def genRule(Item,minConf=0.7):
    if len(Item) >= 2:
        for element in itertools.combinations(list(Item),1): #.combinations 创建一个迭代器，返回list(Item)中所有长度为1的子序列
            conf = support_dic[Item] / float(support_dic[Item - frozenset(element)]) #计算置信度
            if  conf >= minConf:
                print(str([Item - frozenset(element)]) + "-------->" + str(element)) # str(),将参数转换成字符串类型
                print(conf)
                genRule(Item - frozenset(element))

if __name__ == '__main__':
    dataSet = loadDataSet()
    result_list = []    #用来存储规则
    Ck = createC1(dataSet)
    print(Ck)
    #频繁生成频繁项集合，直至产生空集
    while True:
        Lk = getLk(dataSet,Ck,0.5)
        if not Lk:
            break
        result_list.append(Lk)  #将每一轮生成的规则集项加入
        Ck = genLk1(Lk)
        if not Ck:
            break
    #输出频繁项及其支持度
    print("support_dic=",support_dic)
    #输出频繁项集合
    print("result_list=",result_list)
    #输出规则及置信度
    genItem(result_list)
