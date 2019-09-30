import nn
import sys
import numpy as np
import _pickle as cPickle
from util import readMNIST, readCIFAR10
from layers import *

################# TEST CASE 1 #################
def test_case_1():
	X1 = np.asarray([2, 3]).reshape(1,2)

	nn1 = nn.NeuralNetwork(2, 0.0, 1, 1)
	nn1.addLayer(FullyConnectedLayer(2,1))
	nn1.addLayer(FullyConnectedLayer(1,2))

	studActivations = nn1.feedforward(X1)

	# f=open("testcases/testcase1.pkl", 'wb')
	# cPickle.dump(studActivations, f)
	# f.close()

	f=open("testcases/testcase1.pkl", 'rb')
	trueActivations = cPickle.load(f)
	f.close()

	print("Your output :", studActivations)
	print("Expected output : ", trueActivations)

	correct = True

	if len(studActivations) != len(trueActivations):
		correct = False
		print("Shape of the 'returned activations' did not match")

	if correct and type(studActivations) != type(trueActivations):
		correct = False
		print("Type of returned 'activations' is incorrect")

	if correct:
		for i in range(len(trueActivations)):
			if np.allclose(trueActivations[i], studActivations[i]) and studActivations[i].shape == trueActivations[i].shape:
				pass
			else:
				print("Values Don't Match")
				correct = False
				break

	if correct:
		print("Test Case 1 Passed")
		return True
	else:
		print("Test Case 1 Failed")
		return False

################# TEST CASE 2 #################
def test_case_2():
	XTrain, _, _, _, _, _ = readMNIST()
	XTrain=XTrain[0:10,:]
	XTrain.shape = (10, 784)

	nn2 = nn.NeuralNetwork(10, 0.0, 10, 1)
	nn2.addLayer(FullyConnectedLayer(784,100))
	nn2.addLayer(FullyConnectedLayer(100,10))

	studActivations = nn2.feedforward(XTrain)

	# f=open("testcases/testcase2.pkl", 'wb')
	# cPickle.dump(studActivations, f)
	# f.close()

	f=open("testcases/testcase2.pkl", 'rb')
	trueActivations = cPickle.load(f)
	f.close()

	# print("Your output :", studActivations)
	# print("Expected output : ", trueActivations)

	correct = True

	if len(studActivations) != len(trueActivations):
		correct = False
		print("Shape of the 'returned activations' did not match")

	if correct and type(studActivations) != type(trueActivations):
		correct = False
		print("Type of returned 'activations' is incorrect")

	if correct:
		for i in range(len(trueActivations)):
			if np.allclose(trueActivations[i], studActivations[i]) and studActivations[i].shape == trueActivations[i].shape:
				pass
			else:
				print("Values Don't Match")
				correct = False
				break

	if correct:
		print("Test Case 2 Passed")
		return True
	else:
		print("Test Case 2 Failed")
		return False

################# TEST CASE 3 #################
def test_case_3():
	XTrain, _, _, _, _, _ = readCIFAR10()
	XTrain = XTrain[0,:,:,:]
	XTrain.shape = (1,3,32,32)
	nn2 = nn.NeuralNetwork(10, 0.0, 1, 1)
	nn2.addLayer(ConvolutionLayer([3,32,32], [10,10], 32, 2))
	nn2.addLayer(AvgPoolingLayer([32,12,12], [4,4], 4))
	nn2.addLayer(FlattenLayer())
	nn2.addLayer(FullyConnectedLayer(288,100))
	nn2.addLayer(FullyConnectedLayer(100,10))

	studActivations = nn2.feedforward(XTrain)

	# f=open("testcases/testcase3.pkl", 'wb')
	# cPickle.dump(studActivations, f)
	# f.close()

	f=open("testcases/testcase3.pkl", 'rb')
	trueActivations = cPickle.load(f)
	f.close()


	# print("Your output :", studActivations)
	# print("Expected output : ", trueActivations)

	correct = True

	if len(studActivations) != len(trueActivations):
		correct = False
		print("Shape of the 'returned activations' did not match")

	if correct and type(studActivations) != type(trueActivations):
		correct = False
		print("Type of returned 'activations' is incorrect")

	if correct:
		for i in range(len(trueActivations)):
			if np.allclose(trueActivations[i], studActivations[i]) and studActivations[i].shape == trueActivations[i].shape:
				pass
			else:
				print("Values Don't Match")
				correct = False
				break

	if correct:
		print("Test Case 3 Passed")
		return True
	else:
		print("Test Case 3 Failed")
		return False

if __name__ == "__main__":
	np.random.seed(42)
	
	test_case_1()
	test_case_2()
	test_case_3()
