#! /usr/bin/python

import random

S = 5
A = 4
gamma = 0.9

randomSeed = 0

random.seed(randomSeed)

print "numStates",S
print "numActions",A
start = random.randint(0, S-1)
print "start",start
end = random.sample(range(S), random.randint(1, S-1))
print "end",' '.join(map(str,end))

for s in range(0, S):
    for a in range(0, A):
        if s != end:
            degree = random.randint(1,min(5,S))
            l = []
            for i in range(S):
                l.append(i)
            random.shuffle(l)
            R = []
            T = []
            sum = 0
            for i in range(degree):
                x = random.random()
                T.append(x)
                sum += x
                R.append(random.uniform(-1,1))
            for i in range(degree):
                print "transition",s,a,l[i],R[i],T[i]/sum

print "discount ",gamma
