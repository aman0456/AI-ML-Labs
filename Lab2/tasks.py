import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	nn1 = nn.NeuralNetwork(2, 0.075, 100, 30)
	nn1.addLayer(FullyConnectedLayer(2,6))
	nn1.addLayer(FullyConnectedLayer(6,2))
	
	# TASK 2.1 - YOUR CODE HERE
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, True, True)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	nn1 = nn.NeuralNetwork(2, 0.1, 100, 30)
	nn1.addLayer(FullyConnectedLayer(2,2))
	nn1.addLayer(FullyConnectedLayer(2,2))
	# TASK 2.1 - YOUR CODE HERE
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, True, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	nn1 = nn.NeuralNetwork(10, 0.1, 100, 100)
	nn1.addLayer(FullyConnectedLayer(784,20))
	# nn1.addLayer(FullyConnectedLayer(10,10))
	nn1.addLayer(FullyConnectedLayer(20,10))
	# TASK 2.1 - YOUR CODE HERE
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, True, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	
	XTrain = XTrain[0:5000,:,:,:]
	XVal = XVal[0:1000,:,:,:]
	XTest = XTest[0:1000,:,:,:]
	YVal = YVal[0:1000,:]
	YTest = YTest[0:1000,:]
	YTrain = YTrain[0:5000,:]
	
	modelName = 'model.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	nn1 = nn.NeuralNetwork(10, 0.1, 8, 15)
	nn1.addLayer(ConvolutionLayer([3,32,32], [16,16], 32, 2))
	nn1.addLayer(AvgPoolingLayer([32,9,9], [3,3], 2))
	nn1.addLayer(FlattenLayer())
	nn1.addLayer(FullyConnectedLayer(32*4*4,10))
	nn1.addLayer(FullyConnectedLayer(10,10))
	# TASK 2.1 - YOUR CODE HERE
	# raise NotImplementedError
	###################################################
	return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION


	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=False, saveModel=True, modelName=modelName)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
