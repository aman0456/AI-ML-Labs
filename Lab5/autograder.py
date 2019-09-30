import sys, math
import numpy as np
from task import *
from utils import *
import matplotlib.pyplot as plt

def grade1():
	print('='*20 + ' TASK 1 - Preprocessing' + '='*20)
	X = np.arange(6).reshape(3,2).astype(float)
	Y = np.arange(3).reshape(3,1).astype(float)

	try:
		X_stud, Y_stud = preprocess(X, Y)
		X_act, Y_act =  [[ 1., -1.22474487], [ 1., 0.], [ 1.,1.22474487]], [[0.],[1.],[2.]]
		print("Your output :\n", X_stud.shape, Y_stud.shape)

		if np.allclose(X_act, X_stud) and np.allclose(Y_act, Y_stud):
			marks = 1
			print("TASK 1 Passed")
		else:
			marks = 0
			print("Your output : ", X_stud, Y_stud)
			print("Expected output : ", X_act, Y_act)
			print("TASK 1 Failed")
	except KeyError as e:    
		marks = 0
		print("RunTimeError in TASK 1 : " + str(e))
		print("TASK 1 Failed")    
	
	return marks

def grade2(trainX, trainY, testX, testY):
	print('='*20 + ' TASK 2 - Ridge Regression' + '='*20)
	try:
		_lambda = 100
		W_stud = ridge_grad_descent(trainX, trainY, _lambda)
		W_act = np.linalg.inv(trainX.T @ trainX + _lambda * np.eye(trainX.shape[1])) @ trainX.T @ trainY
		if np.linalg.norm(W_stud - W_act) < 1:
			marks = 2
			print("TASK 2 Passed")
		else:
			marks = 0
			print("SSE_Stduent", sse(trainX, trainY, W_stud))
			print("SSE_True", sse(trainX, trainY, W_act))
			print("TASK 2 Failed")

	
	except KeyError as e:    
		marks = 0
		print("RunTimeError in TASK 2 : " + str(e))
		print("TASK 2 Failed")    
	
	return marks


def grade3():
	print('='*20 + ' TASK 3 - K Fold Cross Validation' + '='*20)
	X = np.arange(60).reshape(20,3).astype(float)
	Y = np.arange(20).reshape(20,1).astype(float)
	lambdas = [1,2]
	def dummy(X, Y, _lambda):
		return np.linalg.inv(X.T @ X + _lambda * np.eye(X.shape[1])) @ X.T @ Y
	try:
		scores_stud = k_fold_cross_validation(X, Y, 2, lambdas, dummy)
		scores_act  = [0.12615645, 0.24206764]

		if np.allclose(scores_act, scores_stud):
			marks = 1
			print("TASK 3 Passed")
		else:
			marks = 0
			print("Your output : ", scores_stud)
			print("Expected output : ", scores_act)
			print("TASK 3 Failed")
	except KeyError as e:    
		marks = 0
		print("RunTimeError in TASK 3 : " + str(e))
		print("TASK 3 Failed")  

	return marks  
	
def grade4(trainX, trainY, testX, testY):
	print('='*20 + ' TASK 4 - Lasso Regression' + '='*20)
	try:
		_lambda = 10000
		W_stud = coord_grad_descent(trainX, trainY, _lambda)
		rsse_stud = math.sqrt(sse(trainX, trainY, W_stud))
		rsse_act = 731206
		if rsse_stud - rsse_act < 1e3 and sum(W_stud == 0) > 30:
			marks = 2.5
			print("TASK 4 Passed")
		else:
			marks = 0
			print("Your output : ", rsse_stud)
			print("Expected output : ", rsse_act)
			print("TASK 4 Failed")

	except KeyError as e:    
		marks = 0
		print("RunTimeError in TASK 4 : " + str(e))
		print("TASK 4 Failed")     
	
	return marks

if __name__ == "__main__":
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	if len(sys.argv) < 2:
		print('usage:\npython autograder.py [task-number]\npython autograder.py all')
		sys.exit(1)

	if sys.argv[1].lower() == 'all':
		m1 = grade1()
		m2 = grade2(trainX, trainY, testX, testY)
		m3 = grade3()
		m4 = grade4(trainX, trainY, testX, testY)
		print('='*48 + '\nFINAL GRADE: {}/6.5\n\n'.format(m1 + m2 + m3 + m4))
	elif int(sys.argv[1]) == 1:
		m1 = grade1()
	elif int(sys.argv[1]) == 2:
		m2 = grade2(trainX, trainY, testX, testY)
	elif int(sys.argv[1]) == 3:
		m3 = grade3()
	elif int(sys.argv[1]) == 4:
		m4 = grade4(trainX, trainY, testX, testY)