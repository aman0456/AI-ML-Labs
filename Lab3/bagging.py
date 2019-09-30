import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.legalLabels = legalLabels
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """
        sampleLen = (int)(len(trainingData) * self.ratio)
        self.features = trainingData[0].keys()
        for i in range(self.num_classifiers):
            currTrainData = [trainingData[0]]*sampleLen
            currTrainLabel = [trainingLabels[0]]*sampleLen
            for j in range(sampleLen):
                index = random.randint(0, len(trainingData)-1)
                currTrainData[j] = trainingData[index]
                currTrainLabel[j] = trainingLabels[index]
            self.classifiers[i].train(currTrainData, currTrainLabel)


        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()


    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.
        The function should return a list of labels where each label should be one of legaLabels.
        """

        "*** YOUR CODE HERE ***"
        cnt = len(data)
        ans = [0]*cnt
        
        for i in range(self.num_classifiers):
            temp = self.classifiers[i].classify(data)
            for j in range(cnt):
                ans[j] += temp[j]
        guesses = []
        for elem in ans:
            guess = int(np.sign(elem))
            if guess == 0:
                guess = np.random.choice(self.legalLabels)
            guesses.append(guess)
        # print(data.shape, guesses.shape)
        return guesses
        # util.raiseNotDefined()
