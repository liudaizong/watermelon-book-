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

prob_good = 0.0
prob_bad = 0.0

for i in range(len(data[0])):
	if type(test[i]).__name__ == 'float':
		continue
	else:
		count_good = 0
		count_bad = 0
		for column in range(len(data)):
			if test[i] == data[column,i]:
				if labels[column] == 1:
					count_good += 1
				if labels[column] == 0:
					count_bad += 1
		good = 1.0
		bad = 1.0
		for j in range(len(data[0])):
			if type(test[i]).__name__ == 'float':
				continue
			else:
				good_count = 0
				bad_count = 0
				for column in range(len(data)):
					if test[i] == data[column,i] and test[j] == data[column,j]:
						if labels[column] == 1:
							good_count += 1
						if labels[column] == 0:
							bad_count += 1
				good *= float(good_count + 1) / (count_good + class_number(j))
				bad *= float(bad_count + 1) / (count_bad + class_number(j))
		prob_good += good * (count_good + 1) / float(8 + class_number(i))
		prob_bad += bad * (count_bad + 1) / float(9 + class_number(i)) 

print('probability of good watermelon : %f' % prob_good)
print('probability of bad watermelon : %f' % prob_bad)
if prob_good >= prob_bad:
	print('final result: good watermelon')
else:
	print('final result: bad watermelon')
	