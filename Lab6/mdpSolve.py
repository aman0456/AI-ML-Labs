S = 0
A = 0
st = 0
end = []
transitions = []
discount= 0
values = []
actions = []
def readData(filename):
    global S,A,st,end,transitions,discount,values,actions
    file = open(filename, 'r')
    data = ""
    for inp in file:
        data += inp
    data = data.split()
    # print(data)
    S = int(data[1])
    A = int(data[3])
    st = int(data[5])
    ind = 7
    if int(data[ind]) == -1:
        ind = ind+1
    while str(data[ind]) != "transition":
        end.append(int(data[ind]))
        ind+=1
    for i in range(S):
        transitions.append([])
        for j in range(A):
            transitions[i].append([])
    while(ind < len(data) and str(data[ind]) == "transition"):
        ind+=1
        transitions[int(data[ind])][int(data[ind+1])].append([int(data[ind+2]), float(data[ind+3]), float(data[ind+4])])
        ind += 5
    discount = float(data[ind+1])
    values = [0.0]*S
    actions = [0]*S
    # print(discount, transitions, end, st, S, A)

def value_single_iterate():
    value_new = [0.0]*S
    # print(S, A)
    for i in range(S):
        valList = []
        actionList = []
        for j in range(A):
            if (len(transitions[i][j])) == 0:
            	continue
            y = 0.0
            for k in range(len(transitions[i][j])):
                temp = transitions[i][j][k]
                y += temp[2] * (temp[1] + discount * values[temp[0]])
            valList.append(y)
            actionList.append(j)
        value_new[i] = max(valList) if len(valList) > 0 else 0.0
        actions[i] = actionList[valList.index(value_new[i])] if len(valList) > 0 else 0
    toret = 0.0
    for i in range(S):
        toret = max(toret, abs((values[i] - value_new[i])))
        values[i] = value_new[i]
    return toret

def value_iterate(threshold):
    itercount = 1
    diff = value_single_iterate()
    while diff > threshold:
        diff = value_single_iterate()
        itercount += 1
    return itercount

readData(input())
icount = value_iterate(1e-16)
for i in range(S):
    print(round(values[i], 11), (-1 if (i in end) else actions[i]))
print('iterations', icount)