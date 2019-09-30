import sys
import cluster
import image
import itertools
from math import sqrt


ALLOWED_IMPORTS = [
    'from copy import deepcopy',
    'from itertools import cycle',
    'from pprint import pprint as pprint',
    'from array import array',
    'from cluster import distance_euclidean as distance',
    'import argparse',
    'import sys',
    'import argparse',
    'import matplotlib.pyplot as plt',
    'import random',
    'import math'
]


def compare(a, b):
    if isinstance(a, list):
        same = True
        for i,j in itertools.izip_longest(a,b):
            try:
                same = same and isclose(i,j)
            except Exception:
                same = same and (i==j)
    else:
        try:
            same = isclose(a,b)
        except Exception:
            same = (a==b)
    return same


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def check_imports(filename):
    with open(filename,'r') as f:
        data = f.readlines()

    imports = [i.strip() for i in data if i.find('import') >= 0]
    not_allowed = set(imports) - set(ALLOWED_IMPORTS)
    if len(not_allowed) > 0:
        print('You are not allowed to import anything already present in the code. Please remove the following from {}:'.format(filename))
        for i in not_allowed:
            print(i)
        sys.exit(0)


def grade1():
    print '='*20 + ' TASK 1 ' + '='*20
    testcases = {
        'distance_euclidean': [
            (((2,2), (2,2)), 0),
            (((0,0), (0,1)), 1),
            (((1,0), (0,1)), sqrt(2)),
            (((0,0), (0,-1)), 1),
            (((0,0.5), (0,-0.5)), 1)
        ],
        'kmeans_iteration_one': [
            (([(i,i) for i in range(5)], [(1,1),(2,2),(3,3)], cluster.distance_euclidean), [(0.5, 0.5), (2.0, 2.0), (3.5, 3.5)]),
            (([(i+1,i*2.3) for i in range(5)], [(5,1),(-1,2),(3,6)], cluster.distance_euclidean), [(5, 1), (1.5, 1.15), (4.0, 6.8999999999999995)])
        ],
        'hasconverged': [
            (([(i,i*2,i*3) for i in range(5)], [(i,i*2,i*3+0.01) for i in range(5)], 0.01), True),
            (([(i,i*2,i*3) for i in range(5)], [(i,i*2,i*3+0.01) for i in range(5)], 0.002), False)
        ],
        'iteration_many': [
            (([(i,i) for i in range(3)], [(1,1),(2,2)], cluster.distance_euclidean, 3, 'kmeans', 0.01), [[(1, 1), (2, 2)], [(0.5, 0.5), (2.0, 2.0)], [(0.5, 0.5), (2.0, 2.0)]]),
            (([(i+1,i*2.3) for i in range(3)], [(5,1),(-1,2)], cluster.distance_euclidean, 5, 'kmeans', 0.01), [[(5, 1), (-1, 2)], [(3.0, 4.6), (1.5, 1.15)], [(3.0, 4.6), (1.5, 1.15)]])
        ],
        'performance_SSE': [
            (([(0,i) for i in range(10)], [(0,0), (0,5), (0,10)], cluster.distance_euclidean), 20),
            (([(0,i) for i in range(10)], [(0,0), (0,5), (0,6), (0,10)], cluster.distance_euclidean), 16),
            (([(0,i) for i in range(10)], [(0,50), (-2,5.8), (3,6.1), (0.5,10)], cluster.distance_euclidean), 121.82)
        ]
    }
    grade = 0
    for function in testcases:
        passed = True
        for inp, out in testcases[function]:
            try:
                ret = getattr(cluster,function)(*inp)
            except Exception:
                ret = None
            if not compare(ret, out):
                print 'Function {} failed a testcase.\n\tInput: {}\n\tExpected return: {}\n\tRecieved return: {}\n'.format(function, str(inp)[1:-1], out, ret)
                passed = False
        print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function)
        print '-'*30
        grade += passed

    passed = 1
    for n,k in [(3,3),(10,3),(20,15)]:
        data = [(i,i) for i in range(n)]
        try:
            ret = cluster.initialization_forgy(data, k)
        except Exception:
            ret = None
        if ret is None or len(ret) != k:
            passed = 0
        else:
            passed = sum([1 for i in ret if i not in data]) == 0
        if passed == 0:
            print 'Function initialization_forgy failed a testcase.\n\tInput: {}\n\tExpected return: All cluster centers must come from data points.\n\tRecieved return: {}\n'.format((data, k), ret)
    print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), 'initialization_forgy')
    print '-'*30

    grade = (grade+passed)*0.5

    print 'grade: {}'.format(grade)
    print ''
    return grade


def grade2():
    print '='*20 + ' TASK 2 ' + '='*20
    print 'This task is manually graded. Answer it in the file solutions.txt\n'
    return 0


def grade3():
    print '='*20 + ' TASK 3 ' + '='*20
    grade = 0
    # Test kmeans++.
    for n,k in [(3,3),(10,3),(20,15)]:
        data = [(i,i) for i in range(n)]
        try:
            ret = cluster.initialization_kmeansplusplus(data, cluster.distance_euclidean, k)
        except Exception:
            ret = None
        if ret is None or len(ret) != k:
            passed = 0
        else:
            passed = sum([1 for i in ret if i not in data]) == 0
        if passed == 0:
            print 'Function initialization_kmeansplusplus failed a testcase.\n\tInput: {}\n\tExpected return: All cluster centers must come from data points.\n\tRecieved return: {}\n'.format((data, k), ret)
    print '  {}  Function initialization_kmeansplusplus'.format([u'\u2718', u'\u2713'][passed].encode('utf8'))
    print "NOTE: The autograder doesn't check for correct implementation of this function.\nYour marks depend on whether the TAs are able to understand your code and establish its correctness."
    print '-'*30

    print "\nNOTE: This task has an additional manually graded question worth 2 marks. Answer it in solutions.txt\n"
    print 'grade: {}'.format(grade)
    print ''
    return grade


def grade4():
    print '='*20 + ' TASK 4 ' + '='*20
    testcases = {
        'distance_manhattan': [
            (((2,2), (2,2)), 0),
            (((0,0), (0,1)), 1),
            (((1,0), (0,1)), 2),
            (((0,0), (0,-1)), 1),
            (((2,0.5), (4,-0.5)), 3),
            (((1,0.5), (-1,-0.5)), 3)
        ],
        'kmedians_iteration_one': [
            (([(i,i) for i in range(5)]+[(10,10)], [(1,1),(2,2),(3,3)], cluster.distance_manhattan), [(0.5, 0.5), (2, 2), (4, 4)]),
            (([(i+1,i*2.3) for i in range(4)] + [(-100,-100),(100,100)], [(5,1),(-1,2),(3,6)], cluster.distance_manhattan), [(5, 1), (1, 0.0), (4, 6.8999999999999995)])
        ],
        'performance_L1': [
            (([(0,i) for i in range(10)], [(0,0), (0,5), (0,10)], cluster.distance_manhattan), 12),
            (([(0,i) for i in range(10)], [(0,0), (0,5), (0,6), (0,10)], cluster.distance_manhattan), 10),
            (([(0,i) for i in range(10)], [(0,50), (-2,5.8), (3,6.1), (0.5,10)], cluster.distance_manhattan), 41.2)
        ]
    }
    grade = 0
    for function in testcases:
        passed = True
        for inp, out in testcases[function]:
            try:
                ret = getattr(cluster,function)(*inp)
            except Exception:
                ret = None
            if not compare(ret, out):
                print 'Function {} failed a testcase.\n\tInput: {}\n\tExpected return: {}\n\tRecieved return: {}\n'.format(function, str(inp)[1:-1], out, ret)
                passed = False
        print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function)
        print '-'*30
        grade += passed

    grade *= 0.5

    print 'grade: {}'.format(grade)
    print ''
    return grade


def grade5():
    print '='*20 + ' TASK 5 ' + '='*20
    testcases = {
        'read_image': [
            (['testcases/test_image.pgm'], [map(lambda x: i*x, range(15)) for i in range(1,19)]),
            (['testcases/test_image2.ppm'], [map(lambda x: [i*x, i*x+1, i*x+2], range(15)) for i in range(1,19)])
        ],
        'preprocess_image': [
            ([[map(lambda x: [i*x, i*x+1, i*x+2], range(15)) for i in range(19,1,-1)]], cluster.readfile('testcases/test_data.csv')),
            ([[map(lambda x: [i*x, i*x+1, i*x+2], range(15)) for i in range(1,19)]], cluster.readfile('testcases/test_data2.csv'))
        ]
    }
    grade = 0
    for function in testcases:
        passed = True
        for inp, out in testcases[function]:
            try:
                ret = getattr(image,function)(*inp)
            except Exception:
                ret = None
            if not compare(ret, out):
                print 'Function {} failed a testcase.\n\tInput: {}\n\tExpected return: {}\n\tRecieved return: {}\n'.format(function, str(inp)[1:-1], out, ret)
                passed = False
        print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function)
        print '-'*30
        grade += passed

    grade *= 0.5

    print 'grade: {}'.format(grade)
    print ''
    return grade


def grade6():
    print '='*20 + ' TASK 6 ' + '='*20
    print 'This task is ungraded.\n'
    return 0


def grade7():
    print '='*20 + ' TASK 7 ' + '='*20
    with open('testcases/test_labels', 'r') as f:
        labels = map(lambda x: map(int, x.split()), f.readlines())
    with open('testcases/test_labels2', 'r') as f:
        labels2 = map(lambda x: map(int, x.split()), f.readlines())
    with open('testcases/test_image.pgm', 'r') as f:
        img = f.read()
        img = img.split(None, 4)
    with open('testcases/test_image2.ppm', 'r') as f:
        img2 = f.read()
        img2 = img2.split(None, 4)
    testcases = {
        'label_image': [
            ([[map(lambda x: [i*x, i*x+1, i*x+2], range(15)) for i in range(19,1,-1)], cluster.readfile('testcases/test_centroids.csv')], labels),
            ([[map(lambda x: [i*x, i*x+1, i*x+2], range(15)) for i in range(1,19)], cluster.readfile('testcases/test_centroids2.csv')], labels2)
        ],
        'write_image': [
            (['t1.pgm',[map(lambda x: i*x, range(15)) for i in range(1,19)]], img),
            (['t2.ppm',[map(lambda x: [i*x, i*x+1, i*x+2], range(15)) for i in range(1,19)]], img2)
        ]
    }
    grade = 0
    function = 'label_image'
    passed = True
    for inp, out in testcases[function]:
        try:
            ret = getattr(image,function)(*inp)
        except Exception:
            ret = None
        if not compare(ret, out):
            print 'Function {} failed a testcase.\n\tInput: {}\n\tExpected return: {}\n\tRecieved return: {}\n'.format(function, str(inp)[1:-1], out, ret)
            passed = False
    print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function)
    print '-'*30
    grade += passed

    function = 'write_image'
    passed = True
    for inp, out in testcases[function]:
        try:
            getattr(image,function)(*inp)
            with open(inp[0],'r') as f:
                ret = f.read()
                ret = ret.split(None, 4)
        except Exception:
            ret = None
        if not compare(ret, out):
            print 'Function {} failed a testcase.\n\tInput: {}\n'.format(function, str(inp)[1:-1])
            passed = False
    print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function)
    print '-'*30
    grade += passed

    grade *= 0.5

    print 'grade: {}'.format(grade)
    print ''
    return grade


def grade8():
    print '='*20 + ' TASK 8 ' + '='*20
    with open('testcases/test_labels', 'r') as f:
        labels = map(lambda x: map(int, x.split()), f.readlines())
    with open('testcases/test_labels2', 'r') as f:
        labels2 = map(lambda x: map(int, x.split()), f.readlines())
    with open('testcases/test_decomp', 'r') as f:
        img = [[map(int,j.split(', ')) for j in i.split('], [')] for i in f.read()[3:-3].split(']], [[')]
    with open('testcases/test_decomp2', 'r') as f:
        img2 = [[map(int,j.split(', ')) for j in i.split('], [')] for i in f.read()[3:-3].split(']], [[')]
    testcases = {
        'decompress_image': [
            ([labels, cluster.readfile('testcases/test_centroids.csv')], img),
            ([labels2, cluster.readfile('testcases/test_centroids2.csv')], img2)
        ]
    }
    grade = 0
    function = 'decompress_image'
    passed = True
    for inp, out in testcases[function]:
        try:
            ret = getattr(image,function)(*inp)
        except Exception:
            ret = None
        if not compare(ret, out):
            print 'Function {} failed a testcase.\n\tInput: {}\n\tExpected return: {}\n\tRecieved return: {}\n'.format(function, str(inp)[1:-1], out, ret)
            passed = False
    print '  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function)
    print '-'*30
    grade += passed

    grade *= 0.5

    print 'grade: {}'.format(grade)
    print ''
    return grade


def gradeall(loc):
    print '='*48 + '\nFINAL GRADE: {}\n\n'.format(sum([loc['grade' + str(i)]() for i in range(1,9)]))


for filename in ['cluster.py', 'image.py']:
    check_imports(filename)

if len(sys.argv) < 2:
    print 'usage:\npython autograder.py [task-number]\npython autograder.py all'
    sys.exit(1)
print ''
if sys.argv[1].lower() == 'all':
    gradeall(locals())
else:
    locals()['grade' + str(int(sys.argv[1]))]()
