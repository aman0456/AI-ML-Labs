import numpy as np
import gzip
import _pickle as cPickle

def oneHotEncodeY(Y, nb_classes):
	# Calculates one-hot encoding for a given list of labels
	# Input :- Y : An integer or a list of labels
	# Output :- Coreesponding one hot encoded vector or the list of one-hot encoded vectors
	return (np.eye(nb_classes)[Y]).astype(int)

def readMNIST():
	f = gzip.open('datasets/mnist.pkl.gz', 'rb')
	train_set, val_set, test_set = cPickle.load(f, encoding='latin1')
	f.close()

	trainX, trainY = train_set
	valX, valY = val_set
	testX, testY = test_set
	trainX = np.where(trainX>0, 1, 0)
	valX = np.where(valX>0, 1, 0)
	testX = np.where(testX>0, 1, 0)

	XTrain = np.array(trainX)
	YTrain = np.array(oneHotEncodeY(trainY, 10))
	XVal = np.array(valX)
	YVal = np.array(oneHotEncodeY(valY, 10))
	XTest = np.array(testX)
	YTest = np.array(oneHotEncodeY(testY, 10))

	return XTrain, YTrain, XVal, YVal, XTest, YTest

def readCIFAR10():
	XTrain = np.zeros((40000,3,32,32));
	XVal = np.zeros((10000,3,32,32));
	XTest = np.zeros((10000,3,32,32));
	
	YTrain = np.zeros((40000,1), dtype=np.int8);
	YVal = np.zeros((10000,1), dtype=np.int8);
	YTest = np.zeros((10000,1), dtype=np.int8);
	
	for i in range(1,5):
		with open('./datasets/cifar-10/data_batch_' + str(i), mode='rb') as f:
			data = cPickle.load(f, encoding='bytes')
			XTrain[(i-1)*10000:i*10000, :,:,:] = np.asarray(data[b'data']).reshape(10000,3,32,32)
			YTrain[(i-1)*10000:i*10000,0] = data[b'labels']
	with open('./datasets/cifar-10/data_batch_5', mode='rb') as f:
		data = cPickle.load(f, encoding='bytes')
		XVal[:,:,:,:] = np.asarray(data[b'data']).reshape(10000,3,32,32)
		YVal[:,0] = data[b'labels']
	
	with open('./datasets/cifar-10/test_batch', mode='rb') as f:
		data = cPickle.load(f, encoding='bytes')
		XTest[:,:,:,:] = np.asarray(data[b'data']).reshape(10000,3,32,32)
		YTest[:,0] = data[b'labels']# print(train_set)
	
	YTrain = np.array(oneHotEncodeY(YTrain, 10))
	YVal = np.array(oneHotEncodeY(YVal, 10))
	YTest = np.array(oneHotEncodeY(YTest, 10))
	
	XTrain /=255
	XVal /=255
	XTest /=255
	

	YTrain = np.reshape(YTrain, (40000,10))
	YVal = np.reshape(YVal, (10000,10))
	YTest = np.reshape(YTest, (10000,10))
	
	return XTrain, YTrain, XVal, YVal, XTest, YTest

def split(X,Y):
	Y=Y.astype(int)
	perm = np.arange(X.shape[0])
	np.random.shuffle(perm)
	X = X[perm]
	Y = Y[perm]
	
	trainX = X[0:8000]
	trainY = Y[0:8000]
	XTrain = np.array(trainX)
	YTrain = np.array(oneHotEncodeY(trainY,2))
	
	valX = X[8000:9000]
	valY = Y[8000:9000]
	XVal = np.array(valX)
	YVal = np.array(oneHotEncodeY(valY,2))

	testX = X[9000:10000]
	testY = Y[9000:10000]
	XTest = np.array(testX)
	YTest = np.array(oneHotEncodeY(testY,2))

	return XTrain, YTrain, XVal, YVal, XTest, YTest

def readSquare():
	f=open('datasets/square.pkl', 'rb')
	X, Y = cPickle.load(f)
	f.close()

	return split(X,Y)



def readSemiCircle():
	f=open('datasets/semicircle.pkl','rb')
	X, Y = cPickle.load(f)
	f.close()

	return split(X, Y)