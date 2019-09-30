import matplotlib.pyplot as plt
import numpy as np


def generateGraph(b, filename):
	a = list(range(len(b)))
	plt.xlabel('hill climb iterations')
	plt.ylabel('tour length')
	plt.plot(a, b)
	plt.savefig(filename)


def generateFinalGraph(data, filename, task):
	plt.xlabel('hill climb iterations')
	plt.ylabel('tour length')
	handle = []
	if task == 4:
		plot_euclidean(data)
	else:
		if task == 2:
			label_str = "rseed="
		if task == 3:
			label_str = "starting city="

		for i in range(len(data)):
			a = list(range(len(data[i])))
			h, = plt.plot(a, data[i], label=label_str + str(i+1))
			handle.append(h)

		plt.legend(handles=handle)
	
	plt.savefig(filename)
