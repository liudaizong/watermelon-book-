import numpy as np
import math
import matplotlib.pyplot as plt

#data
x = np.array([0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719])
y = np.array([0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103])
label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

mean1 = np.array([np.mean(x[0:8]), np.mean(x[8:17])]) 
mean2 = np.array([np.mean(x[0:8]), np.mean(y[8:17])]) 

sw = np.zeros((2, 2))
for i in range(8):
	xsmean = np.array([x[i],y[i]]) - np.array([mean1[0],mean2[0]])
	sw += np.dot(xsmean.T, xsmean)
for i in range(8, 17, 1):
	xsmean = np.array([x[i],y[i]]) - np.array([mean1[1],mean2[1]])
	sw += np.dot(xsmean.T, xsmean)

U,S,V = np.linalg.svd(sw)
weight = np.dot(np.dot(V.T, 1/S), U.T)
print weight[1]

#plot
line_x = np.arange(0.1, 0.9, 0.1)
line_y = np.array((-weight[0]*line_x)/weight[1])
p1 = plt.scatter(x[0:8], y[0:8], color = 'g', marker = 'o', label = 'good')
p2 = plt.scatter(x[8:17], y[8:17], color = 'r', marker = 'x', label = 'bad')
plt.plot(line_x, line_y)
plt.xlabel('density')
plt.ylabel('sugar content')
plt.legend()
plt.show()