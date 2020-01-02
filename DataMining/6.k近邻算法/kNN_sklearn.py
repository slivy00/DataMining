"""
@file:kNN_sklearn.py
@author:姚水林
@time:2018-12-19 13:12:45
@function:
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors

def kNN_iris():
    iris = load_iris()     # 加载数据
    X = iris.data[:, :2]    # 为方便画图，仅采用数据的其中两个特征
    y = iris.target
    print(iris)
    # print(iris.feature_names)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')    # 初始化分类器对象,uniform: 统一的权重
    clf.fit(X, y)

    # 画出决策边界，用不同颜色表示
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),#步长0.02
                         np.arange(y_min, y_max, 0.02))#生成以某点为中心指定半径内的

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = 15, weights = 'uniform')")
    plt.show()

def kNN_ball_tree():
    # X = np.array([[-1,-1],
    #               [-2,-1],
    #               [-3,-2],
    #               [1,1],
    #               [2,1],
    #               [3,2]])
    # nbrs = NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(X)
    # distances,indices = nbrs.kneighbors(X)
    # print("distances=",distances)
    # print("indices=",indices)
    # print(nbrs.kneighbors_graph(X).toarray())
    X = [[0],[1],[2],[3]]
    y = [0,0,1,1]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X,y) #训练模型
    print("predict:",neigh.predict([[1.1]])) #预测

if __name__ == '__main__':
    kNN_ball_tree()












