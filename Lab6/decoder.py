import numpy as np
S = 0
A = 4
st = 0
end = []
transitions = []
discount= 2/3
values = []
actions = [0,1,2,3]

def readData(filename, pfilename, p):
	global S,A,st,end,transitions,discount,values,actions
	file = open(filename, 'r')
	pfile = open(pfilename, 'r')
	data = ""
	height = 0
	width = 0

	for inp in file:
		data += inp
		height += 1
	data = data.split()
	width = len(data)//height
	assert(height * width == len(data))
	data = [int(x) for x in data]
	st=0
	end=0
	for i in range(height):
		for j in range(width):
			ind = i*width + j
			if data[ind] == 0:
				data[ind] = S
				S+=1
			elif data[ind] == 2:
				st = S
				data[ind] = S
				S+=1
			elif data[ind] == 3:
				end = S
				data[ind] = S
				S += 1
			else:
				data[ind] = -1
	stateTransitions = {}
	for i in range(S):
		transitions.append([])
		for j in range(A):
			transitions[i].append([])
	for i in range(height):
		for j in range(width):
			ind = i*width + j
			if data[ind] == end:
				continue
			if data[ind] != -1:
				curstate = data[ind]
				validActions = []
				if i > 0 and data[ind - width] != -1:
					validActions.append(0)
					transitions[curstate][0] = data[ind-width]
				if j > 0 and data[ind - 1] != -1:
					validActions.append(3)
					transitions[curstate][3] = data[ind-1]
				if i < height-1 and data[ind + width] != -1:
					validActions.append(2)
					transitions[curstate][2] = data[ind+width]
				if j < width-1 and data[ind + 1] != -1:
					validActions.append(1)
					transitions[curstate][1] = data[ind+1]
				stateTransitions[data[ind]] = validActions
	cnt = 0
	for inp in pfile:
		if cnt != end and cnt < S:
			stateTransitions[cnt] = [int(inp.split()[1])]+stateTransitions[cnt]
		cnt +=1
	actionDist = [-width, 1, width, -1]
	ansstr = []
	def getDir(x):
		if x == 0:
			return 'N'
		elif x == 1:
			return 'E'
		elif x == 2:
			return 'S'
		else:
			return 'W'
	while st != end:
		lent = len(stateTransitions[st])-1
		vprob = (1 - p)/(lent)
		x= np.random.choice(stateTransitions[st], p=[p]+[(1-p)/lent]*lent)
		ansstr.append(getDir(x))
		st = transitions[st][x]
	print(' '.join(ansstr))
gridfilename, policyfilename, p = input().split()
readData(gridfilename, policyfilename, float(p))