import numpy as np
import pandas as pd

def read_data(filename):
	'''
	Reads the input training data from filename and 
	Returns the matrices X : [N X D] and Y : [N X 1] where D is number of features and N is the number of data points
	'''
	dataframe = pd.read_csv(filename, keep_default_na=False, na_values='')
	data = [np.array(dataframe[col]) for col in dataframe]
	for i, d in enumerate(data):
		data[i].shape = (data[i].shape[0], 1)
	data = np.concatenate(data, axis = 1)
	X = data[:,:-1]
	Y = data[:,-1]
	Y.shape = (Y.shape[0],1)
	return X, Y

def one_hot_encode(X, labels):
	'''
	X = input [N X 1] matrix data 
	labels = list of all possible labels for current category
	Returns the matrix X : [N X len(labels)] in one hot encoded format
	'''
	X.shape = (X.shape[0], 1)
	newX = np.zeros((X.shape[0], len(labels)))
	label_encoding = {}
	for i, l in enumerate(labels):
		label_encoding[l] = i
	for i in range(X.shape[0]):
		newX[i, label_encoding[X[i,0]]] = 1
	return newX


def sse(X, Y, W):
	'''
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	W = weight vector [D X 1]
	Returns the Sum Square Error for given set of weights W  
	'''
	return np.linalg.norm(Y - X @ W) ** 2


def ridge_objective(X, Y, W, _lambda):
	'''
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	W = weight vector [D X 1]
	_lambda = scalar parameter
	Returns the Ridge Loss function for given set of weights W  
	'''
	return sse(X,Y,W) + _lambda * np.linalg.norm(W, ord=2)

def lasso_objective(X, Y, W, _lambda):
	'''
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	W = weight vector [D X 1]
	_lambda = scalar parameter
	Returns the Lasso Loss function for given set of weights W  
	'''
	return sse(X,Y,W) + _lambda * np.linalg.norm(W, ord=1)

def separate_data(X, Y):
	'''
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Segregate some part as train and some part as test
	Return the trainX, trainY, testX, testY
	'''
	trainX = X[0:1200, :]
	trainY = Y[0:1200, :]
	
	testX = X[1200:, :]
	testY = Y[1200:, :]

	return trainX, trainY, testX, testY

def plot_kfold(lambdas, scores):
	'''
	lambdas = list of scalar parameter lambda
	scores  = list of average SSE values for each lambda in lambdas
	Plots the kfold cross validation graph required b/w lambdas and scores 
	'''
	plt.plot(lambdas, scores)
	plt.ylabel('Validation_SSE')
	plt.xlabel('Lambda')
	plt.show()
