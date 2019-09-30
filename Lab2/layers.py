import numpy as np
class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.add(np.matmul(X, self.weights),np.tile(self.biases, (n, 1)))
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size
		temp = derivative_sigmoid(self.data) * delta
		toret = np.matmul(temp, np.transpose(self.weights))
		self.weights -= lr * np.matmul(np.transpose(activation_prev), temp)
		self.biases -= lr * np.sum(temp, axis=0)
		# print(toret)
		return toret
		###############################################
		# TASK 2 - YOUR CODE HERE
		raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.numfilters X self.outputsize[0] X self.outputsize[1]]
		self.data = np.zeros(shape=(n, self.out_depth, self.out_row, self.out_col))
		for j in range(self.out_depth):
			jweights = self.weights[j]
			for k in range(self.out_row):
				ks = self.stride*k
				for l in range(self.out_col):
					ls = self.stride*l
					self.data[:,j,k,l] = np.sum(jweights * X[:,:,ks:ks+self.filter_row,ls:ls+self.filter_col], axis=(1,2,3)) + self.biases[j]
		return sigmoid(self.data)
		###############################################
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size
		temp = derivative_sigmoid(self.data) * delta
		toret = np.zeros(shape=(n, self.in_depth, self.in_row, self.in_col))
		for i in range(n):
			for j in range(self.out_depth):
				jweights = self.weights[j]
				for k in range(self.out_row):
					ks = self.stride*k
					for l in range(self.out_col):
						ls = self.stride*l
						toret[i,:,ks:ks+self.filter_row,ls:ls+self.filter_col] += jweights*temp[i][j][k][l]
		for i in range(n):
			xfirst = activation_prev[i]
			for j in range(self.out_depth):
				for k in range(self.out_row):
					ks = self.stride*k
					for l in range(self.out_col):
						ls = self.stride*l
						self.weights[j] -= lr*xfirst[:,ks:ks+self.filter_row,ls:ls+self.filter_col]*temp[i][j][k][l]
		self.biases -= lr*np.sum(temp, axis=(0,2,3))
		return toret
		###############################################
		# TASK 2 - YOUR CODE HERE
		raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.in_channels[2]]
		toret = np.zeros([n,self.out_depth,self.out_row,self.out_col])
		for k in range(self.out_row):
			for l in range(self.out_col):
				toret[:,:,k,l] = np.mean(X[:,:,self.stride*k:self.stride*k+self.filter_row,self.stride*l:self.stride*l+self.filter_col], axis=(2,3))
		return toret

		###############################################
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size
		temp = delta
		toret = np.zeros([n,self.in_depth,self.in_row,self.in_col])
		area = self.filter_row * self.filter_col
		for i in range(n):
			for j in range(self.out_depth):
				for k in range(self.out_row):
					ks = self.stride*k
					for l in range(self.out_col):
						ls = self.stride*l
						toret[i,j,ks:ks+self.filter_row,ls:ls+self.filter_col] += temp[i][j][k][l]/area
		###############################################
		# TASK 2 - YOUR CODE HERE
		return toret
		raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))