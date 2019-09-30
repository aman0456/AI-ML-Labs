import matplotlib.pyplot as plt
import numpy as np
from math import *
from util import *

def draw(X,pred):
	pos_index=np.where(pred==1)
	neg_index=np.where(pred==0)

	X1=X[pos_index]
	X2=X[neg_index]
	plt.plot(X1[:,0], X1[:,1], linestyle="", marker="o", markersize="2",c='r')
	plt.plot(X2[:,0], X2[:,1], linestyle="", marker="o", markersize="2", c='g')

def drawSquare(X,pred):
	draw(X,pred)
	plt.plot([0,0],[-1,1], c='b')
	plt.plot([-1,1],[0,0], c='b')
	plt.show()

def drawSemiCircle(X,pred):
	draw(X,pred)
	circle=plt.Circle((0,0),2, color='b', linestyle="-", fill=False)
	plt.gcf().gca().add_artist(circle)
	plt.show()






