#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import copy as cp
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_9_1.csv')
data = df.values[:, 1:]
index = np.array([idx for idx in range(len(data))])
print index
data = np.column_stack((data,index))
numsample, dim = data.shape
e = 0.11
minpoints = 5

def norm(vec1, vec2):
	dist = np.linalg.norm(vec1 - vec2)
	return dist

kernel_points = np.array([0.0,0.0,0])
queue = {}
all_neibor = {}
for column_i in data:
	count = 0
	list_add = []
	for column_j in data:
		if norm(column_i[:-1], column_j[:-1]) <= e:
			list_add.append(column_j[-1])
			count += 1
	all_neibor[column_i[-1]] = list_add
	if count >= minpoints:
		queue[column_i[-1]] = list_add
		kernel_points = np.vstack((kernel_points, column_i))
print queue
# print all_neibor
kernel_points = np.delete(kernel_points,0,axis=0)
print kernel_points
kernel_points_num = len(kernel_points)

def divide(object_kernel, list_append, kernel_points):
	queue_initial = queue[object_kernel[-1]]
	for value in queue_initial:
		if value in kernel_points[:, -1]:
			print value
			# print list_append
			# print kernel_points
			# print len(kernel_points)
			for i in queue[value]:
				list_append.append(i)
				list_append = list(set(list_append))
			
			for column in range(len(kernel_points)):
				if kernel_points[column, -1] == value:
					row = column
			next_kernel_points = kernel_points[row, :]
			kernel_points = np.delete(kernel_points, row, axis=0)
			list_append = divide(next_kernel_points, list_append, kernel_points)
	return list_append		

k = 0
clss = {}
while len(kernel_points) != 0:
	kernel_index = int(np.random.uniform(0, len(kernel_points)))
	object_kernel = kernel_points[kernel_index, :]
	list_app = []
	# print object_kernel
	list_append = divide(object_kernel, list_app, kernel_points)
	delete_kernel = []
	for i in range(len(kernel_points)):
		if kernel_points[i, -1] in list_append:
			delete_kernel.append(i)
	kernel_points = np.delete(kernel_points, delete_kernel, axis=0)
	clss[k] = list_append
	k += 1

data_last = cp.deepcopy(data)
data_delete = []
for i in range(len(data)):
	for j in clss.keys():
		if i in clss[j]:
			data_delete.append(i)
data_last = np.delete(data_last, data_delete, axis = 0)

#plot
mark = ['or', 'ob', 'og', 'oy']
for column in data_last:
	plt.plot(column[0], column[1], 'ok')
for column in clss.keys():
	markIndex = (clss.keys()).index(column)
	for num_clss in clss[column]:
		plt.plot(data[int(num_clss), 0], data[int(num_clss), 1], mark[markIndex])
plt.xlabel('density')
plt.ylabel('suggar_ratio')
plt.show()