#!python3

import numpy as np

def unitStepFunction(x):
	return np.where(x > 0, 1, 0)

class Perceptron:
	def __init__(self, learningRate=0.02, numberOfIterations=1024):
		self.learningRate = learningRate
		self.numberOfIterations = numberOfIterations
		self.activationFunction = unitStepFunction
		self.weights = None
		self.bias = None
	
	def predict(self, x):
		linearOutput = np.dot(x, self.weights) + self.bias
		yPredicted = self.activationFunction(linearOutput)
		return yPredicted
	
	def fit(self, x, y):
		numSamples, numFeatures = x.shape
		
		# init parameters
		self.weights = np.random.rand(numFeatures) # better if filled with random
		self.bias = 0
		
		# Apply step function to constrain boolean classification
		y_ = np.where(y > 0, 1, 0)
		
		# learn weights
		for _ in range(self.numberOfIterations):
			for index, x_i in enumerate(x):
				linearOutput = np.dot(x_i, self.weights) + self.bias
				yPredicted = self.activationFunction(linearOutput)
				
				# Apply update rule
				update = self.learningRate * (y_[index] - yPredicted)
				self.weights += update * x_i
				self.bias += update

# Testing
if __name__ == "__main__":
	# Imports
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn import datasets
	
	def accuracy(yTrue, yPred):
		accuracy = np.sum(yTrue == yPred) / len(yTrue)
		return accuracy
	
	x, y = datasets.make_blobs(
		n_samples = 150, n_features = 2, centers = 2, cluster_std=1.05, random_state=2
	)
	xTrain, xTest, yTrain, yTest = train_test_split(
		x, y, test_size=0.2, random_state=135
	)
	
	p = Perceptron()
	p.fit(xTrain, yTrain)
	predictions = p.predict(xTest)
	
	print("accuracy:", np.sum(yTest == predictions) / len(yTest))
	
	# Graph decision boundary
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	plt.scatter(xTrain[:, 0], xTrain[:, 1], marker='o', c=yTrain)
	
	x0_1 = np.amin(xTrain[:,0])
	x0_2 = np.amax(xTrain[:,0])
	
	x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
	x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
	
	ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
	
	ymin = np.amin(xTrain[:,1])
	ymax = np.amax(xTrain[:,1])
	ax.set_ylim([ymin - 3, ymax + 3])
	
	plt.show()
