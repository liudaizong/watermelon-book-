import numpy as np 
import math
import pandas as pd
import operator

df = pd.read_csv('watermelon_5_3.csv')
data = df.values[:, 1:-1].tolist()
labels = df.values[:,-1].tolist()
# print data
# print labels

input_layer = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
hidden_layer = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
output_layer = np.array([[0.0]])
hidden_weight = np.random.rand(len(input_layer), len(hidden_layer))
# print hidden_weight.shape
hidden_threshold = np.random.rand(len(hidden_layer),1)
output_weight = np.random.rand(len(hidden_layer), len(output_layer))
output_threshold = np.random.rand(len(output_layer),1)

old_loss = 0.0
step = 0
same_time = 0
lr = 1
while (1):
	cur_loss = 0.0
	delta_output_weight = np.zeros([9,1])
	delta_output_threshold = np.zeros([1,1])
	delta_hidden_weight = np.zeros([8,9])
	delta_hidden_threshold = np.zeros([9,1])
	for column in range(len(data)):
		for i in range(len(input_layer)):
			input_layer[i][0] = data[column][i]
		alpha = np.dot(hidden_weight.T, input_layer) - hidden_threshold
		for i in range(len(hidden_layer)):
			hidden_layer[i] = 1 / (1 + math.exp(-alpha[i]))
		belta = np.dot(output_weight.T, hidden_layer) - output_threshold
		output_layer = 1 / (1 + math.exp(-belta))
		cur_loss += (labels[column] - output_layer) * (labels[column] - output_layer) / 2.0

		g = output_layer * (1 - output_layer) * (labels[column] - output_layer)
		delta_output_weight += np.dot(g, hidden_layer)
		delta_output_threshold += - g

		e = hidden_layer * (1 - hidden_layer) * np.dot(g, output_weight)
		delta_hidden_weight += np.dot(input_layer, e.T)
		delta_hidden_threshold += - e

	cur_loss = cur_loss / len(data)

	output_weight += lr * delta_output_weight
	output_threshold += lr * delta_output_threshold
	hidden_weight += lr * delta_hidden_weight
	hidden_threshold += lr * delta_hidden_threshold

	if abs(cur_loss - old_loss) < 0.0001:
		same_time += 1
		if same_time == 50:
			break
	else:
		old_loss = cur_loss
		same_time = 0
		step += 1
		print('delta time : %d' % step)
 
 #test
predict = []
for column in range(len(data)):
	for i in range(len(input_layer)):
		input_layer[i][0] = data[column][i]
	alpha = np.dot(hidden_weight.T, input_layer) - hidden_threshold
	for i in range(len(hidden_layer)):
		hidden_layer[i] = 1 / (1 + math.exp(-alpha[i]))
	belta = np.dot(output_weight.T, hidden_layer) - output_threshold
	predicti = 1 / (1 + math.exp(-belta))
	predict.append(predicti)

print predict
