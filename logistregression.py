import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv('watermelon_3_3.csv')
data = df.values[:, 1:].tolist()
labels = [example[-1] for example in data]
good_x = [example[0] for example in data[0:8]]
good_y = [example[1] for example in data[0:8]]
bad_x = [example[0] for example in data[8:17]]
bad_y = [example[1] for example in data[8:17]]

old_l = 0.0
weight = np.zeros(3)
while (1):
	cur_l = 0.0
	bx = np.zeros(len(data))
	for i in range(len(data)):
		bx[i] = np.dot(weight, np.array([[data[i][0]], [data[i][1]], [1]]))[0]
		xi =  1 /(1 + math.exp(-bx[i]))
		cur_l -= labels[i] * math.log(xi) + (1 - labels[i]) * math.log(1 - xi) 
	if abs(cur_l - old_l) < 0.001:
		break
	old_l = cur_l
	d1 = 0.0
	lr = 0.03
	for i in range(len(data)):
		xi =  1 /(1 + math.exp(-bx[i]))
		d1 += (xi - labels[i]) * np.array([data[i][0], data[i][1], 1])
	weight -= d1 * lr

xy = [[example[0], example[1], 1] for example in data]
# print np.dot(xy, weight.T)
print labels

#test
predict = []
acc = 0
for i in range(17):
	predicti = 1 - 1/(1+math.exp((np.dot(weight, np.array([[data[i][0]], [data[i][1]], [1]])))[0]))
	if (predicti >= 0.5 and labels[i] == 1) or (predicti < 0.5 and labels[i] == 0):
		acc += 1
	predict.append(predicti)
print predict
Acc = float(acc) / float(len(labels))
print ('Accuracy : %f' % Acc)

#plot
line_x = np.arange(0.1, 0.9, 0.1)
line_y = np.array((-weight[0]*line_x-weight[2])/weight[1])
p1 = plt.scatter(good_x, good_y, color = 'g', marker = 'o', label = 'good')
p2 = plt.scatter(bad_x, bad_y, color = 'r', marker = 'x', label = 'bad')
plt.plot(line_x, line_y)
plt.xlabel('density')
plt.ylabel('sugar content')
plt.legend()
plt.show()

