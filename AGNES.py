import numpy as np 
import pandas as pd 
from math import sqrt
import copy as cp
import matplotlib.pyplot as plt

def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

df = pd.read_csv('watermelon_9_1.csv')
data = df.values[:, 1:]
q = data.shape[0]

C = {}
for i in range(data.shape[0]):
	C[i] = [i]
# print C

M = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
	for j in range(data.shape[0]):
		if i == j:
			M[i, j] = 100.0
		else:
			M[i,j] = norm(data[i, :], data[j, :])

# print np.unravel_index(M.argmin(), M.shape)

while q > 5:
	index_i, index_j = 	np.unravel_index(M.argmin(), M.shape)
	use_j = cp.deepcopy(index_j)
	for i in C[index_j]:
		C[index_i].append(i)
	while True:
		if use_j + 1 == len(C):
			break
		else:
			C[use_j] = C[use_j+1]
			use_j += 1
	del C[use_j]
	for column in range(M.shape[0]):
		if M[column, index_i] < M[column, index_j]:
			M[column, index_i] = M[column, index_j]
	for row in range(M.shape[1]):
		if M[index_i, row] < M[index_j, row]:
			M[index_i, row] = M[index_j, row]
	M = np.delete(M, index_j, axis = 0)
	M = np.delete(M, index_j, axis = 1)
	q -= 1

mark_data = ['or', 'ob', 'og', 'ok', 'oy']
for i in range(q):
	data_list = C[i]
	for j in data_list:
		plt.plot(data[j, 0], data[j, 1], mark_data[i])
plt.show()