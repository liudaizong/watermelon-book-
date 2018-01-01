import numpy as np 
import math
import pandas as pd
import operator

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

hidden_num = 10
p = np.random.rand(10,1)
output = np.random.rand(1,1)
weight = np.random.rand(hidden_num,1)
belta = np.random.rand(hidden_num,1)
lr = 0.5
c = np.random.rand(hidden_num,2)

step = 0
same_time = 0
old_loss = 0.0

while (1):
	step += 1
	cur_loss = 0.0
	dw = np.zeros([hidden_num,1])
	db = np.zeros([hidden_num,1])
	for i in range(len(x)):
		for j in range(hidden_num):
			p[j,0] = math.exp(-belta[j,0] * sum((x[i,:] - c[j,:]) * (x[i,:] - c[j,:])))
		output[0,0] = np.dot(p.T, weight)
		for j in range(hidden_num):
			dw[j,0] += (output[0,0] - y[i,0]) * p[j,0]
			db[j,0] -= (output[0,0] - y[i,0]) * weight[j,0] * sum((x[i,:] - c[j,:]) * (x[i,:] - c[j,:])) * p[j,0]
		cur_loss += (output[0,0] - y[i]) * (output[0,0] - y[i])
	cur_loss = cur_loss / len(x)

	weight -= lr * dw
	belta -= lr * db

	if abs(old_loss - cur_loss) < 0.0001:
		same_time += 1
		if same_time == 10:
			break
	else:
		old_loss = cur_loss
		same_time = 0
	print('delta time : %d' % step)

 #test
predict = []
for i in range(len(x)):
	for j in range(hidden_num):
		p[j,0] = math.exp(-belta[j,0] * sum((x[i,:] - c[j,:]) * (x[i,:] - c[j,:])))
	predicti = np.dot(p.T, weight)
	predict.append(predicti)

print predict
print y.T

	

