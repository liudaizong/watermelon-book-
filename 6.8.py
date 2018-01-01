from sklearn import svm
import pandas as pd
import numpy as np

df = pd.read_csv('watermelon_3_3.csv')
density = df.values[:, 1].tolist()
sugar_ratio = df.values[:, 2].tolist()
data = np.zeros([len(density),1])
labels = np.zeros([len(sugar_ratio),1])
for i in range(len(density)):
	data[i,0] = density[i]
	labels[i,0] = sugar_ratio[i]
# print data
# print labels

clf = svm.SVR()
clf.fit(data, labels)

result = clf.predict(data)
print labels.tolist()
print result.tolist()