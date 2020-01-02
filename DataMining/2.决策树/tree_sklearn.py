"""
@file:tree_sklearn.py
@author:Slivy
@time:2018-12-19 15:17:46
@function:
"""
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

def tree_iris():
    iris = load_iris()
    print("iris=",iris)
    clf = tree.DecisionTreeClassifier()
    train = clf.fit(iris.data,iris.target)
    print("predict=",train.predict([[6.4, 2.8, 5.6, 2.2]]))
    dot_data = tree.export_graphviz(train,out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('./pdf/iris.pdf')

if __name__ == '__main__':
    tree_iris()