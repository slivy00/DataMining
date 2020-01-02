"""
@file:ID3_sklearn.py
@author:姚水林
@time:2018-12-18 20:58:10
@function:
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC,export_graphviz

data = pd.read_csv('./titanic_data.csv',encoding='utf-8')
print("data=",data)
data.drop(['PassengerId'],axis=1,inplace=True)# 舍弃ID列，不适合作为特征
# 数据是类别标签，将其转换为数，用1表示男，0表示女。
data.loc[data['Sex'] == 'male','Sex'] = 1
data.loc[data['Sex'] == 'female','Sex'] = 0
data.fillna(int(data.Age.mean()),inplace=True)#填补空值，inplace=True：原DataFrame中修改
print(data.head(5))

X = data.iloc[:,1:3] #未考虑最后age那一列
y = data.iloc[:,0]


dtc = DTC(criterion='entropy')#ID3算法，使用香农熵
dtc.fit(X,y)
print('输出准确率：', dtc.score(X,y))

# 可视化决策树，导出结果是一个dot文件，需要安装Graphviz才能转换为.pdf或.png格式
with open('./dot/tree.dot','w') as f:
    f = export_graphviz(dtc,feature_names=X.columns,out_file=f)










