import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:-1]
label = df.values[:, -1]

continueornot = []
for column in data[0]:
	if type(column).__name__ == 'str':
		continueornot.append(1)
	else:
		continueornot.append(0)
for i in range(len(continueornot)):
	if continueornot[i] == 0:
		min_value = min(data[:, i])
		max_value = max(data[:, i])
		for j in data:
			j[i] = (j[i] - min_value) / (max_value - min_value)

def compare1(str1, str2):
	if str1 == str2:
		return 0.0
	else:
		return 1.0

def compare2(num1, num2):
	return (num1 - num2) * (num1 - num2)

dist = np.zeros((len(data), len(data)))
for i in range(len(data)):
	for j in range(len(data)):
		for k in range(len(continueornot)):
			if continueornot[k] == 1:
				dist[i, j] += compare1(data[i, k], data[j, k])
			else:
				dist[i, j] += compare2(data[i, k], data[j, k])

weight = np.zeros(len(continueornot))
for i in range(len(data)):
	neibor = dist[i, :]
	nh = 0
	nm = 0
	min_nh = 100.0
	min_nm = 100.0
	for j in range(len(neibor)):
		if i == j:
			continue
		if neibor[j] < min_nh and label[j] == label[i]:
			nh = j
			min_nh = neibor[j]
		if neibor[j] < min_nm and label[j] != label[i]:
			nm = j
			min_nm = neibor[j]
	for k in range(len(continueornot)):
		if continueornot[k] == 1:
			weight[k] = weight[k] - compare1(data[i, k], data[nh, k]) + compare1(data[i, k], data[nm, k])
		else:
			weight[k] = weight[k] - compare2(data[i, k], data[nh, k]) + compare2(data[i, k], data[nm, k])

class_list = {}
for i in range(len(continueornot)):
	class_list[df.columns[i]] = weight[i]
print class_list
class_sort = sorted(class_list.items(), key=lambda item:item[1])
print class_sort
print weight