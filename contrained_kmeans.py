#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

def sort(vec):
	clss = {}
	for i in range(len(vec)):
		clss[vec[i]] = i
	vec_sort = sorted(vec)
	index = []
	for i in range(len(vec)):
		index.append(clss[vec_sort[i]])
	return vec_sort, index


df = pd.read_csv('watermelon_9_1.csv')
data = df.values[:, 1:]
m, n = data.shape
k = 3

centroids = np.vstack((data[6 - 1,:],data[12 - 1,:],data[27 - 1,:]))
c1 = {}

#必连标记，如果样本没有约束，则为0；如果有，则为必连集合编号
flag_c = np.zeros(m,dtype=int)
c1[0] = [4 - 1, 25 - 1]
c1[1] = [12 - 1, 20 - 1] 
c1[2] = [14 - 1 , 17 - 1]
for i in range(k):
	flag_c[c1[i][0]] = i
	flag_c[c1[i][1]] = i

#勿连标记，记录样本是否有勿连标记，如果没有则为0；如果有，则为勿连序列对应的起始序列号
flag_m = np.zeros(m,dtype=int)
m1 = np.array([[2,13,19,21,23,23],[21,23,23,2,13,19]])
m1 -= 1
for i in range(6):
	if flag_m[m1[0,i]] == 0:
		flag_m[m1[0,i]] = i

while True:
	flag_p = np.zeros(m,dtype=int)
	for i in range(m):
		if flag_p[i] > 0:
			continue

		if flag_c[i] > 0:
			tx = (data[c1[flag_c[i]][0],:] + data[c1[flag_c[i]][0],:]) / 2
		else:
			tx = data[i,:]

		dist = np.zeros(k)
		for j in range(k):
			dist[j] = norm(tx, centroids[j,:])
		dist_sort, index = sort(dist)

		for j in range(k):
			tj = index[j]
			ptr = flag_m[i]
			mf = 0
			#如果是勿连约束
			while ptr >0 and ptr < 7 and m1[1, ptr] == i:
				if flag_p[m1[2,ptr]] == tj:
					mf = 1
					break
				ti = m1[2,ptr]
				tdist = norm(data[ti,:], centroids[tj,:])
				if tdist < dist_sort[j]:
					mf = 1
					break
				ptr += 1
			if mf == 1:
				continue
			break

		if mf == 1 and j==k:
			for j in range(k):
				tj = index[j]
				ptr = flag_m[i]
				mf = 0
				while ptr >0 and ptr < 7 and m1[1,ptr] == i:
					if flag_p[m1[2,ptr]] == tj:
						mf = 1
						break
					ptr += 1
				if mf == 1:
					continue
				break

		if flag_c[i] > 0:
			flag_p[c1[flag_c[i]][0]] = tj
			flag_p[c1[flag_c[i]][1]] = tj
		else:
			flag_p[i] = tj

	c = np.zeros((k,30),dtype=int)
	nums = np.zeros(k,dtype=int)
	for i in range(m):
		c[flag_p[i],nums[flag_p[i]]] = i
		nums[flag_p[i]] += 1
	centroids_new = np.zeros((k,2))
	for i in range(k):
		if nums[i] == 0:
			continue
		for j in range(nums[i]):
			centroids_new[i,:] += data[c[i,j],:]
		centroids_new[i,:] /= nums[i]

	delt = norm(centroids_new,centroids)
	if delt < 0.001:
		break
	else:
	 	centroids = centroids_new

#plot
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
for i in range(k):  
	markIndex = i 
	for j in range(nums[i]):
		column = c[i,j]
		plt.plot(data[column, 0], data[column, 1], mark[markIndex]) 

mark = ['+r', '+b', '+g', '+k', '^b', '+b', 'sb', 'db', '<b', 'pb']  
for i in range(k):  
	plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12) 

for i in range(3):
	plt.plot([data[c1[i][0],0],data[c1[i][1],0]], [data[c1[i][0],1],data[c1[i][1],1]],'-y')
	plt.plot([data[m1[0,i],0],data[m1[1,i],0]], [data[m1[0,i],1],data[m1[1,i],1]],'--y')
plt.xlabel('density')
plt.ylabel('suggar_ratio')
plt.show()