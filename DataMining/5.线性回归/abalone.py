"""
@file:abalone.py
@author:姚水林
@time:2018-12-16 16:02:01
@function:
"""
import regression
import matplotlib.pyplot as plt

abX,abY = regression.loadDataSet('abalone.txt')
yHat01 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
ressError01 = regression.rssError(abY[0:99],yHat01.T)
ressError1 = regression.rssError(abY[0:99],yHat1.T)
ressError10 = regression.rssError(abY[0:99],yHat10.T)
print("ressError01=",ressError01,"ressError1=",ressError1,"ressError10=",ressError10)

ridgeWeights = regression.ridgeTest(abX,abY)
print("ridgeWeights=",ridgeWeights)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()





