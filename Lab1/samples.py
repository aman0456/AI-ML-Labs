# samples.py
# ----------
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

import util

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

# Data processing, cleanup and display functions
def loadDataFile(filename, n):
    fin = open(filename).readlines()
    items = [x.rstrip().split("\t") for x in fin[:min(n, len(fin))]]
    items = [[float(y) for y in x] for x in items]
    return items

import zipfile
import os
def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if(os.path.exists(filename)):
        return [l[:] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')

def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels

# Testing
def _test():
    import doctest
    doctest.testmod() # Test the interactive sessions in function comments
    n = 100000
    items = loadDataFile("data/training_data", n)
    labels = loadLabelsFile("data/training_labels", n)
    for i in range(1):
        print items[i]
        print dir(items[i])
    print len(items)
    print len(labels)

if __name__ == "__main__":
    _test()