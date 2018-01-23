#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

'''
1.对每一个存在勿连约束的样本，检查他的所有勿连约束样本，如果已经分类，则正常进行勿连约束判断分类。
如果未分类，则将所有勿连样本拿出来与当前各分类中心进行距离计算，找出一个的分类，使得当前样本与该类
的中心的距离小于所有勿连样本与它的距离。如果有多个这样的分类，选择中心距离当前样本最近的一个。 
2. 如果不存在这样的分类，也正常进行勿连约束判断分类。
'''

#计算两个向量之间的欧式距离
def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

#对一个列表进行排列，默认返回从小到大的排序，并返回对应的索引
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

#初始化中心点
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
	#总的标记类别
	flag_p = np.zeros(m,dtype=int)
	for i in range(m):
		#如果已经标记了类别，就跳过
		if flag_p[i] > 0:
			continue

		#如果存在必连标记，则该点取必连两点的中间值，使得两点能落在同一类中
		if flag_c[i] > 0:
			tx = (data[c1[flag_c[i]][0],:] + data[c1[flag_c[i]][0],:]) / 2
		else:
			tx = data[i,:]

		#计算一点到三个中心点的距离
		dist = np.zeros(k)
		for j in range(k):
			dist[j] = norm(tx, centroids[j,:])
		dist_sort, index = sort(dist)

		#根据勿连标记对该点进行分类
		for j in range(k):
			tj = index[j]	#tj为该类的索引
			ptr = flag_m[i]	#ptr为是否为勿连标记，如果不是则为0，如果是则为勿连下标值
			mf = 0			#错误标记，0为正确，1为错误
			#如果该点是勿连约束
			while ptr >0 and ptr < 7 and m1[1, ptr] == i:
				#如果它的勿连点的类跟自己的类是同一个类，则标记错误
				if flag_p[m1[2,ptr]] == tj:
					mf = 1
					break
				#如果它勿连的类比自己到中心点的距离还近，则为错误
				ti = m1[2,ptr]
				tdist = norm(data[ti,:], centroids[tj,:])
				if tdist < dist_sort[j]:
					mf = 1
					break
				# ptr += 1
			#如果该类报错，则跳转尝试下一个类别
			if mf == 1:
				continue
			#最终出来的值要么是正确的值，即tj代表该点的分类，要么是仍然有错的勿连标记
			break
		#如果对于该勿连点三个类都存在错误信息，那么只进行常规的判断，只需要在不同类即可，不再考虑距离
		if mf == 1 and j==k:
			for j in range(k):
				tj = index[j]
				ptr = flag_m[i]
				mf = 0
				while ptr >0 and ptr < 7 and m1[1,ptr] == i:
					if flag_p[m1[2,ptr]] == tj:
						mf = 1
						break
					# ptr += 1
				if mf == 1:
					continue
				break

		#最后对该点所属类进行划分，根据上面的tj值
		if flag_c[i] > 0:
			flag_p[c1[flag_c[i]][0]] = tj
			flag_p[c1[flag_c[i]][1]] = tj
		else:
			flag_p[i] = tj

	#进行最后的统计，所有点划分后，更新中心点的位置，比较前后中心点的差异
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