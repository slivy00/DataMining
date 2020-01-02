"""
@file:bayes_sklearn.py
@author:姚水林
@time:2018-12-19 13:48:01
@function:
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
print(iris)
clf = GaussianNB()
clf.fit(iris.data,iris.target)
y_pred = clf.predict(iris.data)

print("Number of mislabeled points out of a total %d points : %d"%(iris.data.shape[0],(iris.target != y_pred).sum()))


















