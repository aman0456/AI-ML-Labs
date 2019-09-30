# perceptron1v1.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation
import util
import numpy as np
import sys
import random
PRINT = True

np.random.seed(42)
class Perceptron1v1Classifier:
	"""
	Perceptron classifier.

	Note that the variable 'datum' in this code refers to a counter of features.
	Note that this time around you need k * (k-1) / 2 weight vectors (Counters). 
	The data structure used as keys to these weights is "frozenset". Read up on this.
	"""

	def __init__( self, legalLabels, max_iterations):
		self.legalLabels = legalLabels
		self.type = "perceptron1v1"
		self.max_iterations = max_iterations
		self.weights = {}
		for label1_ind in range(len(legalLabels)):
			for label2_ind in range(label1_ind + 1, len(legalLabels)):
				self.weights[frozenset([label1_ind, label2_ind])] = util.Counter() 

	def setWeights(self, weights):
		assert len(weights) == len(self.legalLabels);
		self.weights = weights;

	def train( self, trainingData, trainingLabels, validationData, validationLabels, validate):
		"""
		The training loop for the perceptron passes through the training data several
		times and updates the weight vector for each label based on classification errors.
		See the assignment description for details.

		Use the provided self.weights[frozenset(label1, label2)] data structure so that
		the classify method works correctly. Also, recall that a
		datum is a counter from features to values for those features
		(and thus represents a vector of values).

		Note *** IMPORTANT ***
		Convention followed for voting algorithm is as follows : 
				if weights[frozenset(label1, label2)] * datum > 0 -> vote max(label1, label2)
				else weights[frozenset(label1, label2)] * datum <= 0 -> vote min(label1, label2)
		"""

		self.features = trainingData[0].keys() # could be useful later
		# DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
		# THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

		f = open("perceptron1v1Iterations.csv","w")
		f_tr = open("perceptron1v1IterationsTrain.csv", "w")
		for iteration in range(self.max_iterations):
			print "Starting iteration ", iteration, "..."
			for i in range(len(trainingData)):
				if(validate):
					if (i % (len(trainingData)/2) ==0):
						guesses = self.classify(validationData)
						correct = [guesses[j] == validationLabels[j] for j in range(len(validationLabels))].count(True)
						f.write(str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")  
						guesses = self.classify(trainingData)
						correct = [guesses[j] == trainingLabels[j] for j in range(len(trainingLabels))].count(True)
						f_tr.write(str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")                      
					

				# "*** YOUR CODE HERE ***"
                                tdata = trainingData[i]
                                tlabel = trainingLabels[i]
                                for label1_ind in range(0,tlabel):
                                    temp = self.weights[frozenset([label1_ind, tlabel])]
                                    if temp * tdata <=0 :
                                        self.weights[frozenset([label1_ind, tlabel])] += tdata
                                for label1_ind in range(tlabel+1,len(self.legalLabels)):
                                    temp = self.weights[frozenset([tlabel, label1_ind])]
                                    if temp * tdata >0 :
                                        self.weights[frozenset([label1_ind, tlabel])] -= tdata
				# util.raiseNotDefined()
				
		## Do not edit code below

		if(validate):
			guesses = self.classify(validationData)
			correct = [guesses[j] == validationLabels[j] for j in range(len(validationLabels))].count(True)
			f.write(str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n") 
			guesses = self.classify(trainingData)
			correct = [guesses[j] == trainingLabels[j] for j in range(len(trainingLabels))].count(True)
			f_tr.write(str(self.max_iterations*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")
		                       
		f.close()
		f_tr.close()

	def classify(self, data ):
		"""
		Classifies each datum as the label that most closely matches the prototype vector
		for that label. See the assignment description for details.

		Recall that a datum is a util.counter...
		"""

		guesses = []
		for datum in data:
			votes = util.Counter()
			
			for l in self.legalLabels:
				votes[l] = 0

			for label1_ind in range(len(self.legalLabels)):
				for label2_ind in range(label1_ind + 1, len(self.legalLabels)):	
					
					label1 = self.legalLabels[label1_ind]
					label2 = self.legalLabels[label2_ind]

					if (self.weights[frozenset([label1, label2])] * datum > 0):
						votes[label2] += 1
					else:
						votes[label1] += 1

			guesses.append(votes.argMax())
		
		return guesses
