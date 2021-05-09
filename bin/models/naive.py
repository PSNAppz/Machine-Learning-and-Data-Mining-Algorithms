import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
import random
import os
import sys



def call_function(DATASET_FOLDER='Dataset', DATASET_OUTPUT_FOLDER='Dataset/output', DATA='Morgan.csv'):

	# prepare data
    try:
        trainingSet=[]
        testSet=[]
        accuracy = 0.0
        split = 0.25
        loadDataset("/".join([DATASET_FOLDER, DATA]), split, trainingSet, testSet)
        print('Train set: ' + repr(len(trainingSet)))
        print('Test set: ' + repr(len(testSet)))
        # generate predictions
        predictions=[]
        trainData = np.array(trainingSet)[:,0:np.array(trainingSet).shape[1] - 1]
        columns = trainData.shape[1] 
        X = np.array(trainData).astype(np.float)
        y = np.array(trainingSet)[:,columns].astype(np.float)
        clf = GaussianNB()
        clf.fit(X, y)
        testData = np.array(testSet)[:,0:np.array(trainingSet).shape[1] - 1]
        X_test = np.array(testData).astype(np.float)
        y_test = np.array(testSet)[:,columns].astype(np.float)
        accuracy = clf.score(X_test,y_test)
        accuracy *= 100
        print(clf.predict([[1996,8,27]]))
        print("Accuracy %:",accuracy)
    except Exception as e:
       
        print( "Error: %s" % e )



def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        for y in range(np.array(dataset).shape[1]):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            testSet.append(dataset[x])
	        else:
	            trainingSet.append(dataset[x])
	

