"""
@file:bostonHousePricing.py
@author:姚水林
@time:2018-12-18 19:15:23
@function:
"""
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

def RM_target():
    """
    住宅平均房间数列与最终房价构造x,y
    :return: 画出二维线性图
    """
    boston = load_boston()
    print("boston=",boston)
    print("boston.keys=",boston.keys())
    print("boston.feature_names=",boston.feature_names)

    x = boston.data[:,np.newaxis,5] # 第6个切出并增加维度，住宅平均房间数列
    y = boston.target
    lm = LinearRegression()  # 声明并初始化一个线性回归模型的对象
    lm.fit(x,y) # 拟合模型，或称为训练模型
    print("u'方程的确定性系数(R^2):%.2f"%lm.score(x,y))#拟合优度越大,说明x对y的解释程度越高,观察点在回归直线附近越密集。

    plt.scatter(x,y,color='green')# 显示数据点
    plt.plot(x,lm.predict(x),color='blue',linewidth=3)# 画出回归直线,lm.predict(x):根据predict方法预测的值
    plt.xlabel('Average Number of Rooms per Dwelling (RM)')
    plt.ylabel('Housing Price')
    plt.title('2D Demo of Linear Regression')
    plt.show()


if __name__ == '__main__':
    RM_target()





