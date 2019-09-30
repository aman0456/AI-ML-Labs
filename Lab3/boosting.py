import util
import numpy as np
import sys
import random
from math import log

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
	classifier, data = y
	return classifier.classify(data)

class AdaBoostClassifier:
	"""
	AdaBoost classifier.

	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	
	"""

	def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
		self.legalLabels = legalLabels
		self.boosting_iterations = boosting_iterations
		self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
		self.alphas = [0.0]*self.boosting_iterations

	def train( self, trainingData, trainingLabels):
		"""
		The training loop trains weak learners with weights sequentially. 
		The self.classifiers are updated in each iteration and also the self.alphas 
		"""
		cnt = len(trainingData)
		weights = [1.0/cnt]*cnt
		self.features = trainingData[0].keys()
		for i in range(self.boosting_iterations):
			self.classifiers[i].train(trainingData, trainingLabels, weights)
			error = 0.0
			temp = self.classifiers[i].classify(trainingData)
			for j in range(cnt):
				if temp[j] != trainingLabels[j]:
					error += weights[j]
			for j in range(cnt):
				if temp[j] == trainingLabels[j]:
					weights[j] *= 1.0*error/(1.0-error)
			wsum = sum(weights)
			weights = [k/wsum for k in weights]
			self.alphas[i] = 1.0*(log((1.0-error)/error))
			# print("classifier" , i, "trained")
		# util.raiseNotDefined()

	def classify( self, data):
		"""
		Classifies each datum as the label that most closely matches the prototype vector
		for that label. This is done by taking a polling over the weak classifiers already trained.
		See the assignment description for details.

		Recall that a datum is a util.counter.

		The function should return a list of labels where each label should be one of legaLabels.
		"""
		toret = []
		guesses = [0.0]*len(data)
		for i in range(self.boosting_iterations):
			cguess = self.classifiers[i].classify(data)
			for j in range(len(data)):
				guesses[j] += self.alphas[i] * float(cguess[j])
		for j in range(len(data)):
			guess = int(np.sign(guesses[j]))
			if guess == 0:
				guess = np.random.choice(self.legalLabels)
			toret.append(guess)
		return toret
		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()