"""
@file:predictGlasses.py
@author:姚水林
@time:2018-12-09 14:44:05
@function:预测隐形眼镜类型
"""
import trees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)