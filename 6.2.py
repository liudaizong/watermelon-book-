from sklearn import svm
import pandas as pd

df = pd.read_csv('watermelon_5_3.csv')
data = df.values[:, 1:-1]
labels = df.values[:,-1]
# print data
# print labels

clf = svm.SVC()
clf.fit(data, labels)

result = clf.predict(data)
print result
print labels

print clf.decision_function(data)
print clf.support_vectors_
