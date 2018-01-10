import numpy as np 
import pandas as pd 
from math import sqrt
import matplotlib.pyplot as plt

#data
x = np.array([2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1])
y = np.array([2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9])

#mean and normalization
mean_x = np.mean(x)
mean_y = np.mean(y)
scaled_x = x - mean_x
scaled_y = y - mean_y
data = np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])
plt.plot(scaled_x,scaled_y,'or') 

#Covariance Matrix
cov = np.dot(data.T, data) / len(data)

#svd
eig_val, eig_vec = np.linalg.eig(cov)
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='blue')

#change the world
new_data=np.transpose(np.dot(eig_vec,np.transpose(data)))
plt.plot(new_data[:,0],new_data[:,1],'^',color='blue')

#select
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature = eig_pairs[0][1]

#z
new_data_reduced = (np.dot(feature.T, data.T)).T
plt.plot(new_data_reduced[:,0],[1.2]*10,'*',color='green')
plt.show()
