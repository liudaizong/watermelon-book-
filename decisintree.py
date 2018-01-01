#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import pandas as pd 
from math import log
import operator
from treePlotter import *
'''
dataset: 数据集，每一行为一组数据
value：对应的类的值
feature：属性
'''
#计算信息熵
def Ent(dataSet):
    total_number = len(dataSet)
    labelCounts = {}
    for column in dataSet:
        currentLabel = column[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / total_number
        Ent -= prob * log(prob, 2)

    return Ent

#对离散值的划分
def splitNotContinuousDataSet(dataSet, axis, value):
    retDataset = []
    for column in dataSet:
        if column[axis] == value:
            reducedColumn = column[:axis]
            reducedColumn.extend(column[axis+1:])
            retDataset.append(reducedColumn)

    return retDataset

#对连续值的划分
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataset = []
    for column in dataSet:
        if direction == 0:
            if column[axis] > value:
                # reducedColumn = column[:axis]
                # reducedColumn.extend(column[axis+1:])
                retDataset.append(column)
        else:
            if column[axis] <= value:
                # reducedColumn = column[:axis]
                # reducedColumn.extend(column[axis+1:])
                retDataset.append(column)

    return retDataset

#选择最优的属性划分
def chooseBestFeaturetoSplit(dataSet, features):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = Ent(dataSet)
    bestGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        if type(featureList[0]).__name__ == 'float' or type(featureList[0]).__name__ == 'int':
            sortList = sorted(featureList)
            splitList = []
            for j in range(len(sortList) - 1):
                splitList.append((sortList[j] + sortList[j+1]) / 2.0)
            bestSplitEntropy = 100
            slen = len(splitList)
            for j in range(slen):
                value = splitList[j]
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i ,value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * Ent(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * Ent(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            bestSplitDict[features[i]] = splitList[bestSplit]
            Gain = baseEntropy - bestSplitEntropy
        else:
            uniqueVals = set(featureList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = splitNotContinuousDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob*Ent(subDataSet)
            Gain = baseEntropy - newEntropy
        if Gain > bestGain:
            bestGain = Gain
            bestFeature = i
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[features[bestFeature]]
        features[bestFeature] = features[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] < bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0

    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def creatTree(dataSet, features, data_full, features_full):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeaturetoSplit(dataSet, features)
    bestFeatFeature = features[bestFeat]
    mytree = {bestFeatFeature:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentLabel = features_full.index(features[bestFeat])
        featValuesFull = [example[currentLabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del(features[bestFeat])
    for value in uniqueVals:
        subFeature = features[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        mytree[bestFeatFeature][value] = creatTree(splitNotContinuousDataSet(dataSet, bestFeat, value), subFeature, data_full, features_full)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            mytree[bestFeatFeature][value] = majorityCnt(classList)

    return mytree

df = pd.read_csv('watermelon_3_3.csv')
data = df.values[:, 1:].tolist()
data_full = data[:]
features = df.columns.values[1:-1].tolist()
# print features
features_full = features[:]
myTree = creatTree(data, features, data_full, features_full)
print myTree
createPlot(myTree)
