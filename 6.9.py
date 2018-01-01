import numpy as np
import math
import matplotlib.pyplot as plt

#data
x = np.array([0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719])
y = np.array([0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103])
label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#paramater
old_l = 0.0
step = 0
b = np.zeros([18,1])
b[17,0] = 1

k = np.ones([18,18])
for i in range(17):
	for j in range(17):
		k[i,j] = math.exp(-0.5 * (x[i] - x[j]) * (x[i] - x[j]))

while 1:
	cur_l = 0.0
	bx = np.zeros(17)

	#newton
	# for i in range(17):
	# 	bx[i] = (np.dot(weight, np.array([[x[i]],[y[i]],[1]])))[0]
	# 	cur_l = cur_l + (( -label[i] * bx[i]) + math.log(1 + math.exp(bx[i])))
	
	#gradient
	for i in range(17):
		bx[i] = np.dot(b.T, k[:,i])
		cur_l += -y[i] * bx[i] + math.log(1 + math.exp(bx[i]))

	if abs(cur_l - old_l) < 0.001:
		break

	step += 1
	old_l = cur_l
	p_y1 = np.zeros(17)
	d1 = np.zeros([18,1])
	d2 = np.zeros([18,1])
	#newton
	# for i in range(17):
	# 	p_y1[i] = 1 - 1/(1+math.exp(bx[i]))
	# 	xi = np.array([[x[i]],[y[i]],[1]])
	# 	d1 = d1 - np.dot(label[i] - p_y1[i], xi)
	# 	d2 = d2 + p_y1[i] * ( 1 - p_y1[i] ) * np.dot(xi, xi.T)
	
	#gradient
	for i in range(17):
		p_y1[i] = 1 - 1/(1+math.exp(bx[i]))
		for j in range(18):
			d2[j,0] = k[j,i]
		d1 = d1 - d2 * (y[i] - p_y1[i])
	b -= d1 * 0.01

#test
predict = []
acc = 0
for i in range(17):
	predicti = np.dot(b.T, k[:,i])
	if (predicti >= 0.5 and label[i] == 1) or (predicti < 0.5 and label[i] == 0):
		acc += 1
	predict.append(predicti)
Acc = float(acc) / float(len(label))
print ('Accuracy : %f' % Acc)

#plot
p1 = plt.scatter(x[0:8], y[0:8], color = 'g', marker = 'o', label = 'good')
p2 = plt.scatter(x[8:17], y[8:17], color = 'r', marker = 'x', label = 'bad')
plt.xlabel('density')
plt.ylabel('sugar content')
plt.legend()
plt.show()
