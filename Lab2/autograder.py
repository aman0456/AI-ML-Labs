import sys
import test_feedforward as tf 
import test
import numpy as np

def grade1():
    np.random.seed(42)
    print('='*20 + ' TASK 1 - Forward Pass' + '='*20)
    marks = 0
    try:
        if tf.test_case_1():
            marks += 1
    except:    
        print("RunTimeError in Test Case 1")
    
    try:
        if tf.test_case_2():
            marks += 1
    except:    
        print("RunTimeError in Test Case 2")
    
    try:
        if tf.test_case_3():
            marks += 2
    except:    
        print("RunTimeError in Test Case 3")
        
    
    print('Marks: {}/4'.format(marks))
    return marks

def grade2():
    print('='*20 + ' TASK 2 - Forward + Backward Pass' + '='*20)
    marks = 0
    
    try:
        net, xtest, ytest = test.task[1](False)
        marks += test_net(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.1")

    try:
        net, xtest, ytest = test.task[2](False)
        marks += test_net(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.2")

    try:
        net, xtest, ytest = test.task[3]()
        marks += 3 * test_net(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.3")

    try:
        net, xtest, ytest, name = test.task[4]()
        model = np.load(name)
        k,i = 0,0
        for l in net.layers:
            if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
                net.layers[i].weights = model[k]
                net.layers[i].biases = model[k+1]
                k+=2
            i+=1

        marks += 4 * test_net_diff(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.4")
    

    print('Marks: {}/9'.format(marks))
    return marks


def test_net(net, xtest, ytest):
    _, acc  = net.validate(xtest, ytest)
    if acc >= 90:
        return 1
    elif acc >= 85:
        return 0.75
    elif acc >= 75:
        return 0.5
    else:
        return 0

def test_net_diff(net, xtest, ytest):
    _, acc  = net.validate(xtest, ytest)
    print(acc)
    if acc >= 35:
        return 1
    elif acc >= 30:
        return 0.75
    elif acc >= 25:
        return 0.5
    else:
        return 0

if len(sys.argv) < 3:
    print('usage:\npython3 autograder.py -t task-number')
    sys.exit(1)

locals()['grade' + str(int(sys.argv[2]))]()
