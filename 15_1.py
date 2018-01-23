#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import copy as cp
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_4_2.csv')
data = df.values[:10, 1:]
idx = df.values[:10,0]
data = np.column_stack((data,idx))
labels = df.values[:10, -1]
features = df.columns[1:-1].tolist()
clss = {}
for i in range(len(features)):
	clss[features[i]] = set(data[:,i])
num_samples, dim = data.shape
rules = {}
rule = []
rule_num = 0

def sort_samefeature(clss):
	max_rate = 0.0
	best_value = ''
	count_m = 0
	for i in clss.keys():
		if clss[i][0] > max_rate:
			max_rate = clss[i][0]
			best_value = i 
			count_m = clss[i][1]
	return best_value, max_rate, count_m

def sort_features(clss):
	max_rate = 0.0
	best_index = 0
	best_feature = ''
	best_value = ''
	count_all = 0
	for i in range(len(features)):
		if clss[features[i]][1] > max_rate:
			best_feature = features[i]
			max_rate = clss[features[i]][1]
			best_index = i 
			best_value = clss[features[i]][0]
			count_all = clss[features[i]][2]
		if clss[features[i]][1] == max_rate and clss[features[i]][2] > count_all:
			best_feature = features[i]
			max_rate = clss[features[i]][1]
			best_index = i 
			best_value = clss[features[i]][0]
			count_all = clss[features[i]][2]
		if clss[features[i]][1] == max_rate and clss[features[i]][2] == count_all and i < best_index:
			best_feature = features[i]
			max_rate = clss[features[i]][1]
			best_index = i 
			best_value = clss[features[i]][0]
			count_all = clss[features[i]][2]
	return best_feature, best_value, max_rate

def change(data, feature, value):
	list_change = []
	for i in range(len(data)):	
		if data[i,features.index(feature)] != value:
			list_change.append(i)
	data = np.delete(data, list_change, axis=0)
	data = np.delete(data, features.index(feature), axis=1)
	return data

def right_change(data, feature, value):
	list_change = []
	for i in range(len(data)):
		if data[i,features.index(feature)] != value:
			list_change.append(i)
	data = np.delete(data, list_change, axis=0)
	return data

data_remain = cp.deepcopy(data)
features_remain = cp.deepcopy(features)
while  True:
	rule_num += 1
	while True:
		rule_features = {}
		for i in range(len(features)):
			rule_feature = {}
			for feature_value in clss[features[i]]:
				count_m = 0
				count_n = 0
				for column in range(len(data)):
					if data[column, i] == feature_value:
						count_m += 1
					if data[column, i] == feature_value and data[column,-2] == 1:
						count_n += 1
				if count_m == 0:
					continue
				rate = float(count_n) / count_m
				rule_feature[feature_value] = [rate, count_m]
			best_value, max_rate, count_all = sort_samefeature(rule_feature)
			rule_features[features[i]] = [best_value, max_rate, count_all]
			print rule_feature
		print rule_features
		best_feature, best_value, total_rate = sort_features(rule_features)
		rules[best_feature] = best_value
		print rules
		if total_rate != 1.0:
			data = change(data, best_feature, best_value)
			print data
			features.remove(best_feature)
			print features
		else:
			data = right_change(data, best_feature, best_value)
			print data
			break
	rule.append(rules)
	rules = {}
	features = cp.deepcopy(features_remain)
	list_all = []
	for i in range(len(data)):
		for j in range(len(data_remain)):
			if data_remain[j, -1] == data[i, -1]:
				list_all.append(j)
	print list_all
	data_remain = np.delete(data_remain,list_all,axis=0)
	data = cp.deepcopy(data_remain)
	print data
	if len(data) == (len(labels) - np.count_nonzero(labels)):
		break
print rule

