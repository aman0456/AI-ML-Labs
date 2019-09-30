import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	xshape = X.shape
	retX = np.ones((xshape[0], 1))
	for dimen in range(1, xshape[1]):
		dtype = type(X[0][dimen])
		# print (dtype)
		currCol = X[:,dimen]
		if dtype == type("AA"):
			X1 = one_hot_encode(currCol, list(set(currCol)))
			retX = np.concatenate((retX, X1), axis=1)
		else:
			# print(dtype, currCol)
			normalisedX1 = (currCol - np.mean(currCol))/np.std(currCol)
			# print (retX, normalisedX1.reshape(-1, 1))
			retX = np.concatenate((retX, normalisedX1.reshape(-1, 1)), axis=1)
	return retX.astype(float), Y.astype(float)

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	return -2.0 * np.matmul(X.T, (Y - np.matmul(X, W))) + 2.0 * _lambda * W

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	xshape = X.shape
	myweights = np.random.normal(0,0.001,(xshape[1],1))
	for iterate in range(max_iter):
		# print (myweights)
		grad = grad_ridge(myweights, X, Y, _lambda)
		# print("", grad)
		gradNorm = np.linalg.norm(grad);
		if gradNorm < epsilon:
			return myweights
		myweights -= lr * grad
	return myweights

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	def calcSSE(myX, myY, myW, mylambda):
		# print(myX.shape, myY.shape, myW.shape)
		myt = (myY - np.matmul(myX, myW))
		return float(np.matmul(myt.T, myt))

	xshape = X.shape
	retX = np.array_split(X,k)
	retY = np.array_split(Y,k)
	# print("here3")
	toret = []
	for mylambda in lambdas:
		# print("here4")
		ans = 0.0
		for i in range(k):
			# print("here5")
			toTrainX = np.concatenate(retX[:i] + retX[i+1:])
			toTrainY = np.concatenate(retY[:i] + retY[i+1:])
			# print("here6")
			retWeights = algo(toTrainX, toTrainY, mylambda)

			ans += calcSSE(retX[i], retY[i], retWeights, mylambda)
		ans /=k
		toret.append(ans)
		print(mylambda, ans)
	return toret



def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	xshape = X.shape
	myweights = np.random.normal(0, 0.01, (xshape[1], 1))
	for iterate in range(max_iter):
		val = np.matmul(X.T, X)
		for dimen in range(xshape[1]):
			currCol = X[:,dimen]
			sigma1 = np.sum(currCol*Y[:,0])
			sigma2 = val[dimen][dimen]
			sigma3 = np.sum(val[:, dimen]*myweights[:,0]) - sigma2*myweights[dimen]
			value1 = (2*sigma1 - 2*sigma3 - _lambda)/(2*sigma2)
			
			if value1 > 0:
				myweights[dimen] = value1
			else:
				value2 = (2*sigma1 - 2*sigma3 + _lambda)/(2*sigma2)
				if value2 < 0:
					myweights[dimen] = value2
				else:
					myweights[dimen] = 0
		# print("iter", iterate, "done")
	return myweights

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	# print("here")
	lambdas = [11, 12, 12.5, 13, 13.5, 14, 15] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	lambdas = [(ml - 9) * 1e5 for ml in lambdas]
	# for i in range(20):
		# lambdas.append(i*1)
	# print("here1")
	mylambda = 3.5e5
	res = coord_grad_descent(trainX, trainY, mylambda)
	print(res, sse(testX, testY, res))
	# scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	# plot_kfold(lambdas, scores)