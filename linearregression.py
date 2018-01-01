import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv('watermelon_3_3.csv')
line_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
line_y = [2, 5, 7, 7, 9, 12, 15, 20, 17, 22]

old_l = 0.0
weight = np.zeros(2)
while (1):
	cur_l = 0.0
	bx = np.zeros(len(line_x))
	for i in range(len(line_x)):
		bx[i] = np.dot(weight, np.array([[line_x[i]], [1]]))
		cur_l += (bx[i] - line_y[i]) * (bx[i] - line_y[i])
	cur_l = cur_l / (2 * len(line_x))
	if abs(cur_l - old_l) < 0.001:
		break
	old_l = cur_l
	d1 = 0.0
	lr = 0.001
	for i in range(len(line_x)):
		d1 += (bx[i] - line_y[i]) * np.array([line_x[i], 1])
	weight -= d1 * lr / len(line_x)

predict_y = []
for i in line_x:
	yi = i * weight[0] + weight[1]
	predict_y.append(yi)

p1 = plt.scatter(line_x, line_y, color = 'g', marker = 'o')
plt.plot(line_x, predict_y)
plt.xlabel('density')
plt.ylabel('sugar content')
plt.legend()
plt.show()
