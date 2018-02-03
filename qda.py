import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as LDA
import csv
import random

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	accuracy = 0.0
	split = 0.25
	loadDataset('Dataset/med.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	trainData = np.array(trainingSet)[:,0:np.array(trainingSet).shape[1] - 1]
  	columns = trainData.shape[1] 
	X = np.array(trainData).astype(np.float)
	y = np.array(trainingSet)[:,columns].astype(np.float)
	clf = LDA()
	clf.fit(X, y)
	testData = np.array(testSet)[:,0:np.array(trainingSet).shape[1] - 1]
	X_test = np.array(testData).astype(np.float)
	y_test = np.array(testSet)[:,columns].astype(np.float)
	accuracy = clf.score(X_test,y_test)
	accuracy *= 100
	print("Accuracy %:",accuracy)	




def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        for y in range(np.array(dataset).shape[1]):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            testSet.append(dataset[x])
	        else:
	            trainingSet.append(dataset[x])




def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

main()	
	

