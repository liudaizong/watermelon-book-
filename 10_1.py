import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_3_3.csv')
data = df.values[:, 1:-1]
label = df.values[:,-1].tolist()

def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

k = 3
distance = np.zeros(len(data))
for i in np.arange(0.22, 0.78, 0.02):
	for j in np.arange(0.02, 0.48, 0.02):
		for l in range(len(data)):
			distance[l] = norm(np.array([i, j]), data[l,:])
		arr = sorted(distance)
		vote1 = 0
		vote0 = 0
		for l in range(len(data)):
			if distance[l] in arr[:k]:
				if label[l] == 1:
					vote1 += 1
				else:
					vote0 += 1
			if (vote0 + vote1) == 3:
				break
		if vote0 > vote1:
			plt.plot(i, j, '.y')
		else:
			plt.plot(i, j, '.g')

for l in range(len(data)):
	if label[l] == 0:
		plt.plot(data[l, 0], data[l, 1], 'oy')
	else:
		plt.plot(data[l, 0], data[l, 1], 'og')
plt.xlabel('density')
plt.ylabel('suggar_ratio')
plt.show()

