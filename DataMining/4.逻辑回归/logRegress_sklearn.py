"""
@file:logRegress_sklearn.py
@author:姚水林
@time:2018-12-18 20:19:59
@function:
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression
from sklearn.model_selection import train_test_split #用于训练集和测试集的划分，
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def logRegress():
    """
    对指定数据集进行逻辑回归
    :return: none
    """
    data = pd.read_csv('./LogisticRegression.csv',encoding='utf-8')
    print("data.head(5)=\n",data.head(5)) # 查看数据框的头五行
    '''
    离散特征的编码分为两种情况：
    1、离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
    2、离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}
    '''
    data_dum = pd.get_dummies(data,prefix='rank',columns=['rank'],drop_first=True)
    print("data_dum.tail(5)=\n",data_dum.tail(5))  # 查看数据框的最后五行
    #ix:通过行标签或者行号索引行数据(.ioc;.iloc)
    """
    train_test_split(train_data,train_target,test_size=0.4, random_state=0)
    train_data：所要划分的样本特征集
    train_target：所要划分的样本结果
    test_size：样本占比，如果是整数的话就是样本的数量
    random_state：是随机数的种子。
    """
    X_train,X_test,y_train,y_test = train_test_split(data_dum.iloc[:, 1:], data_dum.iloc[:, 0], test_size=.1, random_state=520)
    # print(X_test)
    lr = LogisticRegression() #建立Lr模型
    lr.fit(X_train,y_train) #训练数据模型
    print('逻辑回归的准确率为：{0:.2f}%'.format(lr.score(X_test,y_test)*100))# str.format()

def linear_model():
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    Y = iris.target
    h = .02
    logreg  = LogisticRegression(C=1e5)#正则化系数，越小正则化程度越高
    logreg.fit(X,Y)
    x_min,x_max = X[:,0].min() - .5,X[:,0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == '__main__':
    # logRegress()
    linear_model()









