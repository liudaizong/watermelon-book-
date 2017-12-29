#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import pandas as pd 
from math import log, exp
import operator

def entropy(dataset):
	ent = 0.0
	countclass = {}
	total_class = [example[-1] for example in dataset]
	for clss in total_class:
		if clss not in countclass.keys():
			countclass[clss] = 0
		countclass[clss] += 1
	for key in countclass:
		prob = float(countclass[key]) / len(total_class)
		ent = ent - prob * log(prob, 2)
	return ent

def splitcontinue(dataset, weight, up):
	returndataset = []
	for column in dataset:
		if up == 0:
			if (column[0] * weight[0][0] + column[1] * weight[0][1])  > - weight[0][2]:
				returndataset.append(column)
		else:
			if (column[0] * weight[0][0] + column[1] * weight[0][1]) <= - weight[0][2]:
				returndataset.append(column)
	return returndataset

def chooseweight(dataset):
	label = [example[-1] for example in dataset]
	weight = [[0, 0, 0]]
	while 1:
		bx = np.zeros(len(dataset))
		for i in range(len(dataset)):
			bx[i] = (np.dot(weight, np.array([[dataset[i][0]],[dataset[i][1]],[1]])))[0]
			yi = 1/(1 + exp(-bx[i])) 
		print weight
		dataset1 = splitcontinue(dataset, weight, 0)
		classlist1 = [example[-1] for example in dataset1]
		dataset2 = splitcontinue(dataset, weight, 1)
		classlist2 = [example[-1] for example in dataset2]
		if len(classlist1) != 0 and len(classlist2) != 0:
			if classlist1.count(classlist1[0]) == len(classlist1) or classlist2.count(classlist2[0]) == len(classlist2):
				break
		p_y1 = np.zeros(len(dataset))
		d1 = 0
		for i in range(len(dataset)):
			p_y1[i] = 1 - 1/(1+exp(bx[i]))
			xi = np.array([[dataset[i][0]],[dataset[i][1]], [1]])
			d1 = d1 - np.dot(label[i] - p_y1[i], xi)
		weight = weight - d1.T* 0.03
	return weight

def vote(classlist):
	vote1 = 0
	vote0 = 0
	for clss in classlist:
		if clss == 1:
			vote1 = vote1 + 1
		else:
			vote0 = vote0 + 1
	if vote1 >= vote0:
		return 1
	else:
		return 0

def createTree(dataset, features, data_full, features_full):
	classlist = [example[-1] for example in dataset]
	if classlist.count(classlist[0]) == len(classlist):
		return classlist[0]
	bestweight = chooseweight(dataset)
	feature_best = str(bestweight[0][0]) + 'x' + features[0] + '+' +  str(bestweight[0][1]) + 'x' + features[1] + '<=' + str(-bestweight[0][2])
	mytree = {feature_best:{}}
	for value in [1, 0]:
		mytree[feature_best][value] = createTree(splitcontinue(dataset, bestweight, value), features, data_full, features_full)
	return mytree

df = pd.read_csv('watermelon_3_3.csv')
data = df.values[:, 1:].tolist()
data_full = data[:]
features = df.columns.values[1:-1].tolist()
features_full = features[:]
myTree = createTree(data, features, data_full, features_full)
print myTree
