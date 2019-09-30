import optparse
import os
import re
import sys

import perceptron
import bagging
import boosting
import samples
import util
import dataClassifier

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def runTask(task):
	print "Grading task "+str(task) 
	if task == 4 :
		print "This is a manually graded task, write your answers in the answers.txt file"
	elif task == 1:
		print "Making sure that you understood the algorithms can't be autograded for the moment. Tune back in 2050 for some exciting advances ;)"
	elif task in [2, 3]:
		numTraining = 1000
		numTest = 1000
		num_classes = 2	
	
		rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
		trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
		trainingLabels = [-1 if 0 <= x <= 4 else 1 for x in trainingLabels]
		rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
		testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
		testLabels = [-1 if 0 <= x <= 4 else 1 for x in testLabels]
		featureFunction = dataClassifier.basicFeatureExtractorDigit
	
		legalLabels = [-1,1]

		if task == 2:
			classifier = bagging.BaggingClassifier(legalLabels,3,perceptron.PerceptronClassifier,1,20)
		if task == 3:
			classifier = boosting.AdaBoostClassifier(legalLabels,3,perceptron.PerceptronClassifier,20)

		# Extract features
		print "Extracting features..."
		trainingData = map(featureFunction, rawTrainingData)
		testData = map(featureFunction, rawTestData)

		# Conduct training and testing
		print "Training..."
		classifier.train(trainingData, trainingLabels)

		print "Testing..."
		guesses = classifier.classify(testData)
		correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
		acc = 100*correct/(1.0*(len(testLabels)))

		if task == 2:
			marks = 0
			if (acc > 75):
				marks = 3
			elif (acc >  73):
				marks = 2
			elif (acc >  71):
				marks = 1	
			print "Received Marks : " + str(marks) +"/3"
		elif task == 3:
			marks = 0
			if (acc > 75):
				marks = 4
			elif (acc >  74):
				marks = 3
			elif (acc >  73):
				marks = 2		
			elif (acc >  71):
				marks = 1
			print "Received Marks : " + str(marks) +"/4"
	print "--------------------------------------------------------"		

if __name__ == '__main__':
	# Read input
	from optparse import OptionParser
	parser = OptionParser(dataClassifier.USAGE_STRING)
	parser.add_option('-t', '--task', help=dataClassifier.default('The task to autograde'), choices=['1', '2', '3', '4'], default=None)
	
	options, otherjunk = parser.parse_args(sys.argv[1:])
	if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

	task_list = [1, 2, 3, 4]
	if options.task:
		runTask(int(options.task))
	else:
		for task in task_list:
			try :
				runTask(task)
			except:
				print "Task" +str(task)	+" exited without running to completion"
				if task == 2:
					number = 3
				if task == 3:
					number = 4
				
				print "Received Marks : " + str(0) +("/%d" % number)
				print "--------------------------------------------------------"

