from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np 

iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
print  scores.mean()   

df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:-1]
labels = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
clf = AdaBoostClassifier(n_estimators=10)
scores = cross_val_score(clf, data, labels)
print scores.mean()