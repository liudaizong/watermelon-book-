#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import pandas as pd 
from math import log
import operator
import copy

def Gini(dataset):
	gini = 1.0
	countclass = {}
	total_class = [example[-1] for example in dataset]
	for clss in total_class:
		if clss not in countclass.keys():
			countclass[clss] = 0
		countclass[clss] += 1
	for key in countclass:
		prob = float(countclass[key]) / len(total_class)
		gini = gini - prob * prob
	return gini

def splitsperate(dataset, id, value):
	returndataset = []
	for column in dataset:
		if column[id] == value:
			reducedata = column[:id]
			reducedata.extend(column[id+1:])
			returndataset.append(reducedata)
	return returndataset

def splitcontinue(dataset, id, value, up = False):
	returndataset = []
	for column in dataset:
		if up :
			if column[id] > value:
				# reducedata = column[:id]
				# reducedata.extend(column[id+1:])
				returndataset.append(column)
		else:
			if column[id] <= value:
				# reducedata = column[:id]
				# reducedata.extend(column[id+1:])
				returndataset.append(column)
	return returndataset

def choosebestfeature(dataset, features):
	num_feature = len(features)
	MinGini = 100
	bestFeature = -1
	bestvalue_all = 0.0
	bestvalue = 0.0
	for i in range(num_feature):
		if type(dataset[0][i]).__name__ == 'float' or type(dataset[0][i]).__name__ == 'int':
			featurevalue = [example[i] for example in dataset]
			splitlist = []
			for j in range(len(featurevalue) - 1):
				splitlist.append((sorted(featurevalue)[j] + sorted(featurevalue)[j+1]) / 2.0)
			mingini = 100
			for value in splitlist:
				newgini = 0.0
				dataset1 = splitcontinue(dataset, i, value)
				prob1 = float(len(dataset1)) / float(len(dataset))
				newgini = newgini + prob1 * Gini(dataset1)
				dataset2 = splitcontinue(dataset, i, value, up = True)
				prob2 = float(len(dataset2)) / float(len(dataset))
				newgini = newgini + prob2 * Gini(dataset2)
				if newgini < mingini:
					mingini = newgini
					bestvalue = value
		else:
			featurevalue = [example[i] for example in dataset]
			values = set(featurevalue)
			mingini = 0.0
			for value in values:
				dataset3 = splitsperate(dataset, i, value)
				prob = float(len(dataset3)) / float(len(dataset))
				mingini = mingini + prob * Gini(dataset3)
		if MinGini > mingini:
			MinGini = mingini
			bestFeature = i
			bestvalue_all = bestvalue
	if type(dataset[0][bestFeature]).__name__ == 'float' or type(dataset[0][bestFeature]).__name__ == 'int':
		features[bestFeature] = features[bestFeature] + '<=' + str(bestvalue_all)
		for i in range(shape(dataset)[0]):
			if dataset[i][bestFeature] <= bestvalue_all:
				dataset[i][bestFeature] = 1
			else:
				dataset[i][bestFeature] = 0
	return bestFeature

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

def classify(inputTree, features_labels, testvec):
	firststr = inputTree.keys()[0]
	if '<=' in firststr:
		value = float(firststr.split('<=')[-1])
		feature_key = firststr.split('<=')[0]
		second_dict = inputTree[firststr]
		feature_index = features_labels.index(feature_key)
		if testvec[feature_index] <= value:
			judge = 1
		else:
			judge = 0
		for key in second_dict.keys():
			if judge == int(key):
				if type(second_dict[key]).__name__ == 'dict':
					classlabel = classify(second_dict[key], features_labels, testvec)
				else:
					classlabel = second_dict[key]
	else:
		second_dict = inputTree[firststr]
		feature_index = features_labels.index(firststr)
		for key in second_dict.keys():
			if testvec[feature_index] == key:
				if type(second_dict[key]).__name__ == 'dict':
					classlabel = classify(second_dict[key], features_labels, testvec)
				else:
					classlabel = second_dict[key]
	return classlabel

def testing(myTree, data_test, labels):
	error = 0.0
	for i in range(len(data_test)):
		if classify(myTree, labels, data_test[i]) != data_test[i][-1]:
			error += 1.0
	return error

def testingMajor(major, data_test):
	error = 0.0
	for i in range(len(data_test)):
		if major != data_test[i][-1]:
			error += 1.0
	return error

def createTree(dataset, features, data_full, features_full, data_test):
	classlist = [example[-1] for example in dataset]
	if classlist.count(classlist[0]) == len(classlist):
		return classlist[0]
	if len(dataset[0]) == 1:
		return vote(classlist)
	features_copy = copy.deepcopy(features)
	feature_best_id = choosebestfeature(dataset, features)
	feature_best = features[feature_best_id]
	mytree = {feature_best:{}}
	featValues = [example[feature_best_id] for example in dataset]
	uniqueVals = set(featValues)
	if type(dataset[0][feature_best_id]).__name__ == 'str':
		currentLabel = features_full.index(features[feature_best_id])
		featValuesFull = [example[currentLabel] for example in data_full]
		uniqueValsFull = set(featValuesFull)
	del(features[feature_best_id])
	for value in uniqueVals:
		subFeature = features[:]
		if type(dataset[0][feature_best_id]).__name__ == 'str':
			uniqueValsFull.remove(value)
		mytree[feature_best][value] = createTree(splitsperate(dataset, feature_best_id, value), subFeature, data_full, features_full, splitsperate(data_test, feature_best_id, value))
	if type(dataset[0][feature_best_id]).__name__ == 'str':
		for value in uniqueValsFull:
			mytree[feature_best][value] = vote(classlist)
	return mytree

def postPruningTree(inputTree, dataset, data_test, features):
	firststr = inputTree.keys()[0]
	second_dict = inputTree[firststr]
	classlist = [example[-1] for example in dataset]
	feature_key = copy.deepcopy(firststr)
	if '<=' in firststr:
		feature_key = firststr.split('<=')[0]
		feature_value = firststr.split('<=')[-1]
	feature_index = features.index(feature_key)
	features_copy = copy.deepcopy(features)
	del(features[feature_index])
	for key in second_dict.keys():
		if type(second_dict[key]).__name__ == 'dict':
			if type(dataset[0][feature_index]).__name__ == 'str':
				inputTree[firststr][key] = postPruningTree(second_dict[key], splitsperate(dataset, feature_index, key), splitsperate(data_test, feature_index, key), copy.deepcopy(features))
			else:
				inputTree[firststr][key] = postPruningTree(second_dict[key], splitcontinue(dataset, feature_index ,feature_value, key), splitcontinue(data_test, feature_index, feature_value, key), copy.deepcopy(features))
	if testing(inputTree, data_test, features_copy) <= testingMajor(vote(classlist), data_test):
		return inputTree
	return vote(classlist)

df = pd.read_csv('watermelon_4_2.csv')
data = df.values[:11, 1:].tolist()
data_full = data[:]
data_test = df.values[11:,1:].tolist()
features = df.columns.values[1:-1].tolist()
features_full = features[:]
myTree = postPruningTree(createTree(data, features, data_full, features_full, data_test), data, data_test, features_full)
print myTree

