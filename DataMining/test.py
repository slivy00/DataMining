import random
import numpy as np

li = [[4.4, 2.9, 1.4, 0.2], [4.5, 2.3, 1.3, 0.3], [4.3, 3.0, 1.1, 0.1], [4.8, 3.0, 1.4, 0.1]]
ran_li = random.sample(li, 3)
print(ran_li)
afterMultiplication = []
for i,x in enumerate(ran_li):
    # print(i,"---",x)
    x = np.array(x) * (1 / (i + 1))
    afterMultiplication.append(x.tolist())
    # print(i, "---", x)
print("afterMultiplication=",afterMultiplication)

f = np.array([[1,2],[3,4]])
print("f=",f,"\nf*f",np.multiply(f,[[1,2]]))
