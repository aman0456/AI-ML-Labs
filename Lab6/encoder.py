S = 0
A = 4
st = 0
end = []
transitions = []
discount= 2/3
values = []
actions = [0,1,2,3]

def readData(filename, p):
	global S,A,st,end,transitions,discount,values,actions
	file = open(filename, 'r')
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
				end.append(S)
				data[ind] = S
				S += 1
			else:
				data[ind] = -1
	for i in range(S):
		transitions.append([])
		for j in range(A):
			transitions[i].append([])
	for i in range(height):
		for j in range(width):
			ind = i*width + j
			if data[ind] == end[0]:
				continue
			if data[ind] != -1:
				curstate = data[ind]
				validActions = []
				if i > 0 and data[ind - width] != -1:
					validActions.append(0)
				if j > 0 and data[ind - 1] != -1:
					validActions.append(3)
				if i < height-1 and data[ind + width] != -1:
					validActions.append(2)
				if j < width-1 and data[ind + 1] != -1:
					validActions.append(1)
				ivprob = (1 - p)/len(validActions)
				vprob = p + ivprob
				actionDist = [-width, 1, width, -1]
				if i > 0 and data[ind - width] != -1:
					transitions[curstate][0].append((data[ind-width],-1.0, vprob))
					for action in validActions:
						if action != 0:
							transitions[curstate][action].append((data[ind-width],-1.0, ivprob))
				if j > 0 and data[ind - 1] != -1:
					transitions[curstate][3].append((data[ind-1],-1.0, vprob))
					for action in validActions:
						if action != 3:
							transitions[curstate][action].append((data[ind-1],-1.0, ivprob))
				if i < height-1 and data[ind + width] != -1:
					transitions[curstate][2].append((data[ind+width],-1.0, vprob))
					for action in validActions:
						if action != 2:
							transitions[curstate][action].append((data[ind+width],-1.0, ivprob))
				if j < width-1 and data[ind + 1] != -1:
					transitions[curstate][1].append((data[ind+1],-1.0, vprob))
					for action in validActions:
						if action != 1:
							transitions[curstate][action].append((data[ind+1],-1.0, ivprob))	
	print("numStates", S)
	print("numActions", 4)
	print("start", st)
	print("end", end[0])
	for i in range(S):
		for j in range(A):
			for trans in transitions[i][j]:
				print("transition", i, j, trans[0], trans[1], trans[2])
	print("discount", discount)

filename,p = input().split()
readData(filename, float(p))