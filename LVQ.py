import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

df = pd.read_csv('watermelon_9_1.csv')
data = df.values[:, 1:]
label = np.ones(data.shape[0], dtype=int)
label[8:21] = 2
q = 5
lr = 0.1 

pvec = np.zeros((q, data.shape[1]))
plabel = [1, 2, 2, 1, 1]
initial_sample = [5, 12, 18, 23, 29]
for i in range(len(initial_sample)):
	pvec[i, :] = data[initial_sample[i] - 1, :]

total_label = np.zeros(data.shape[0], dtype=int)

for time in range(400):
	index_random = int(np.random.uniform(0, data.shape[0]))

	min_dist = 100.0
	min_p = 100
	for i in range(q):
		dist = norm(data[index_random, :], pvec[i, :])
		if dist < min_dist:
			min_dist = dist
			min_p = i 
	if plabel[min_p] == label[index_random]:
		total_label[index_random] = min_p
		pvec[min_p, :] += lr * (data[index_random, :] - pvec[min_p, :])
	else:
		pvec[min_p, :] -= lr * (data[index_random, :] - pvec[min_p, :])


mark_data = ['or', 'ob', 'og', 'ok', 'oy']
for i in range(len(data)):
	mark_index = total_label[i]
	plt.plot(data[i, 0], data[i, 1], mark_data[mark_index])

mark_p = ['+r', '+b', '+g', '+k', '+y']
for i in range(len(pvec)):
	mark_index = i
	plt.plot(pvec[i, 0], pvec[i, 1], mark_p[mark_index], markersize = 12)
plt.show()