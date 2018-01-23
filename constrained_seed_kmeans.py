#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_9_1.csv')
data = df.values[:, 1:]
k = 3

def initCentroids(data, k):
	numsample, dim = data.shape  
	centroids = np.zeros((k, dim))
	centroids[0, :] = data[s1[0], :] + data[s1[1], :]
	centroids[1, :] = data[s2[0], :] + data[s2[1], :]
	centroids[2, :] = data[s3[0], :] + data[s3[1], :]
	print s1
	return centroids / 2 

def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

s1 = [4, 25]
s2 = [12, 20]
s3 = [14, 17]
s_all = [s1, s2, s3] 
centroids = initCentroids(data, k)
while (1):
	c = np.zeros((k,30), dtype=int)
	num_c = np.zeros((k,1), dtype=int)
	for s in range(k):
		c[s, 0] = s_all[s][0]
		c[s, 1] = s_all[s][1]
		num_c[s,0] = 2
	for i in range(30):
		min_distance = 10000
		min_dim = 0
		for j in range(k):
			distance = norm(data[i,:], centroids[j,:])
			if distance < min_distance:
				min_distance = distance
				min_dim = j
		c[min_dim,num_c[min_dim,0]] = i 
		num_c[min_dim,0] += 1

	centroids_new = np.zeros((k,2))
	for i in range(k):
		if num_c[i,0] == 0:
			continue
		for j in range(num_c[i,0]):
			centroids_new[i,:] += data[c[i,j],:]
		centroids_new[i,:] /= num_c[i,0]

	delta = norm(centroids_new, centroids)
	if delta < 0.001:
		break
	else:
		centroids = centroids_new

#plot
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
for i in range(k):  
	markIndex = i 
	for j in range(num_c[i,0]):
		column = c[i,j]
		plt.plot(data[column, 0], data[column, 1], mark[markIndex]) 

mark = ['+r', '+b', '+g', '+k', '^b', '+b', 'sb', 'db', '<b', 'pb']  
for i in range(k):  
	plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12) 
plt.xlabel('density')
plt.ylabel('suggar_ratio')
plt.show()