import numpy as np 
import pandas as pd 
from math import log, exp, pow, sqrt, pi

df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:-1]
test = df.values[0,1:-1]
labels = df.values[:,-1].tolist()

def class_number(index):
	class_number = {}
	for column in data:
		if column[index] not in class_number.keys():
			class_number[column[index]] = 0
		class_number[column[index]] += 1
	num = len(class_number)
	return num

def continue_para(num, index):
	ave = 0.0
	var = 0.0
	count = 0
	for column in range(len(data)):
		if labels[column] == num:
			count += 1
			ave += data[column,index]
	ave = ave / count
	for column in range(len(data)):
		if labels[column] == num:
			var += (data[column,index] - ave) * (data[column,index] - ave)
	var = var / count

	return ave,var

prob_good = log((8 + 1) / float(17 + 2))
prob_bad = log((9 + 1) / float(17 + 2))

for i in range(len(data[0])):
	if type(test[i]).__name__ == 'float':
		ave0, var0 = continue_para(0, i)
		ave1, var1 = continue_para(1, i)
		prob0 = exp(- pow(test[i] - ave0, 2) / (2 * var0)) / sqrt(2 * pi * var0)
		prob1 = exp(- pow(test[i] - ave1, 2) / (2 * var1)) / sqrt(2 * pi * var1)
		prob_good += log(prob1)
		prob_bad += log(prob0)
	else:
		count_good = 0
		count_bad = 0
		for column in range(len(data)):
			if test[i] == data[column,i]:
				if labels[column] == 1:
					count_good += 1
				if labels[column] == 0:
					count_bad += 1
		prob_good += log(float(count_good + 1) / (8 + class_number(i)))
		prob_bad += log(float(count_bad + 1) / (9 + class_number(i)))

print('probability of good watermelon : %f' % prob_good)
print('probability of bad watermelon : %f' % prob_bad)
if prob_good >= prob_bad:
	print('final result: good watermelon')
else:
	print('final result: bad watermelon')
