"""
@file:aprior_sklearn.py
@author:Slivy
@time:2018-12-19 20:35:41
@function:
"""
import  pyfpgrowth
transaction = [
    [1,2,5],
    [2,4],
    [2,3],
    [1,2,4],
    [1,3],
    [2,3],
    [1,3],
    [1,2,3,5],
    [1,2,3]
]
patterns = pyfpgrowth.find_frequent_patterns(transaction,2) #支持度为2，置信度为0.7
rules = pyfpgrowth.generate_association_rules(patterns,0.7)
print("patterns=",patterns)
print("rules=",rules)
