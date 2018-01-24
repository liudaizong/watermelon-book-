#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import copy as cp
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:-3]
idx = df.values[:,0]
label = df.values[:,-1]
data = np.column_stack((data,label))
data = np.column_stack((data,idx))
labels = df.values[:, -1]
features = df.columns[1:-3].tolist()
rules = {}
rule = []
rule_num = 0

def choose_feature(rule_vec, data):
	max_pos = 0
	feature = ''
	index = []
	for i in range(len(rule_vec)):
		count_pos = 0
		count_neg = 0
		index_new = []
		rule_new = cp.deepcopy(rule_vec)
		rule_new.pop(i)
		for j in range(len(data)):	
			list_j = cp.deepcopy(data[j,:-2].tolist())
			list_j.pop(i)
			if list_j == rule_new:
				if data[j, -2] == 1:
					index_new.append(data[j, -1])
					count_pos += 1
				else:
					count_neg += 1
		# print count_pos
		if count_neg != 0:
			continue
		if count_pos > max_pos:
			max_pos = count_pos
			feature = features[i]
			index = index_new
	return feature,index

data_remain = cp.deepcopy(data)
features_remain = cp.deepcopy(features)
while True:
	#if remain label = 0
	count_zero = 0
	for column in data:
		if column[-2] == 0:
			count_zero += 1
	if count_zero == len(data):
		break
	rule_num += 1
	special_rule = data[0, :-2].tolist()
	for i in range(len(special_rule)):
		rules[features[i]] = special_rule[i]
	# print special_rule
	while True:
		feature_delete, index = choose_feature(special_rule, data)
		# print special_rule
		if feature_delete != '':
			feature_value = special_rule[features.index(feature_delete)]
			del rules[feature_delete]
			special_rule.pop(special_rule.index(feature_value))
			data = np.delete(data, features.index(feature_delete), axis=1)
			features.pop(features.index(feature_delete))
			index_current = index
		else:
			break
	print rules
	features = cp.deepcopy(features_remain)
	list_delete = []
	for i in range(len(data_remain)):
		if data[i, -1] in index_current:
			list_delete.append(i)
	data = np.delete(data_remain, list_delete, axis=0)
	data_remain = cp.deepcopy(data)
	rule.append(rules)
	rules = {}