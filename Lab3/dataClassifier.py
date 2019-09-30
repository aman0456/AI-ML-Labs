# dataClassifier.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import perceptron
import bagging
import boosting
import samples
import sys
import util


sys.setrecursionlimit(3000)

TEST_SET_SIZE = 1000
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features =  basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print rawTestData[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c perceptron -t 1000 -f -s 1000
                  - would run the perceptron classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the digits dataset, would test the classifier on the test data of 1000 examples
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['perceptron', 'bagging', 'boosting'], default='bagging')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=1000, type="int")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-v', '--validate', help=default("Whether to validate when training (for graphs)"), default=False, action="store_true")
    parser.add_option('-r', '--ratio', help=default('The ratio of dataset to be used to train 1 weak classifier'), default=0.1, type=float)
    parser.add_option('-n', '--num_classifiers', help=default('The number of weak classifier to be trained on each subset of dataset'), default=10, type=int)
    parser.add_option('-b', '--boosting_iteration', help=default('Maximum iterations to run adaboost algorithm'), default=2, type=int)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "classifier:\t\t" + options.classifier
    if options.classifier == "bagging":
        print "num of weak classifier:%d" % options.num_classifiers
    if options.classifier == "boosting":
        print "num of boosting iterations:%d" % options.boosting_iteration
    print "training set size:\t" + str(options.training)

    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage

    featureFunction = basicFeatureExtractorDigit
    
    legalLabels = [-1, 1]

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if (options.classifier == "perceptron"):
        classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
    elif (options.classifier == "bagging"):
        classifier = bagging.BaggingClassifier(legalLabels,options.iterations,perceptron.PerceptronClassifier,options.ratio,options.num_classifiers)
    elif (options.classifier == "boosting"):
        classifier = boosting.AdaBoostClassifier(legalLabels,options.iterations,perceptron.PerceptronClassifier,options.boosting_iteration)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING
        sys.exit(2)


    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    trainingLabels = [-1 if 0 <= x <= 4 else 1 for x in trainingLabels]
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    validationLabels = [-1 if 0 <= x <= 4 else 1 for x in validationLabels]
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    testLabels = [-1 if 0 <= x <= 4 else 1 for x in testLabels]

    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels)
    guesses = classifier.classify(trainingData)
    correct = [guesses[i] == trainingLabels[i] for i in range(len(trainingLabels))].count(True)
    print str(correct), ("correct out of " + str(len(trainingLabels)) + " (%.1f%%).") % (100.0 * correct / len(trainingLabels))
    
    if(options.classifier == "perceptron"):
        f = open("perceptron_train.csv","a")
        f.write(str(len(trainingData))+","+str(100*correct/(1.0*(len(trainingData))))+'\n')
        f.close()
    elif(options.classifier == "bagging"):
        f = open("bagging_train.csv","a")
        f.write(str(options.num_classifiers)+","+str(100*correct/(1.0*(len(trainingData))))+'\n')
        f.close()
    elif(options.classifier == "boosting"):
        f = open("boosting_train.csv","a")
        f.write(str(options.boosting_iteration)+","+str(100*correct/(1.0*(len(trainingData))))+'\n')
        f.close()

    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    
    if(options.classifier == "perceptron"):
        f = open("perceptron_val.csv","a")
        f.write(str(len(trainingData))+","+str(100*correct/(1.0*(len(validationData))))+'\n')
        f.close()
    elif(options.classifier == "bagging"):
        f = open("bagging_val.csv","a")
        f.write(str(options.num_classifiers)+","+str(100*correct/(1.0*(len(validationData))))+'\n')
        f.close()
    elif(options.classifier == "boosting"):
        f = open("boosting_val.csv","a")
        f.write(str(options.boosting_iteration)+","+str(100*correct/(1.0*(len(validationData))))+'\n')
        f.close()
    
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
    
    if(options.classifier == "perceptron"):
        f = open("perceptron_test.csv","a")
        f.write(str(len(trainingData))+","+str(100*correct/(1.0*(len(testData))))+'\n')
        f.close()
    elif(options.classifier == "bagging"):
        f = open("bagging_test.csv","a")
        f.write(str(options.num_classifiers)+","+str(100*correct/(1.0*(len(testData))))+'\n')
        f.close()
    elif(options.classifier == "boosting"):
        f = open("boosting_test.csv","a")
        f.write(str(options.boosting_iteration)+","+str(100*correct/(1.0*(len(testData))))+'\n')
        f.close()
        

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
