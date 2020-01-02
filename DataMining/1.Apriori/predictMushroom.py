"""
@file:predictMushroom.py
@author:姚水林
@time:2018-12-09 21:15:40
@function:发现毒蘑菇的相似特征
第一个特征表示有毒或者可食用。如果某样本有毒，则值为2。如果可食用，则值为1。
下一个特征是蘑菇伞的形状，有六种可能的值，分别用整数3-8来表示。

"""
import apriori

mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
L,suppData = apriori.apriori(mushDatSet,minSupport=0.3)
for item in L[1]:
    if  item.intersection('2'):print(item)#intersection() 方法用于返回两个或更多集合中都包含的元素，即交集。

