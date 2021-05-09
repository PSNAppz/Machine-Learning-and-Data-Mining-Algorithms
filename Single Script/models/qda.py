import os
import sys
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import csv
import random
from dotenv import load_dotenv

# environment variable
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET')


def call_function():
    try:
        # prepare data
        trainingSet=[]
        testSet=[]
        accuracy = 0.0
        split = 0.25
        loadDataset("/".join([DATASET_FOLDER, 'med.data']), split, trainingSet, testSet)
        # generate predictions
        predictions=[]
        trainData = np.array(trainingSet)[:,0:np.array(trainingSet).shape[1] - 1]
        columns = trainData.shape[1] 
        X = np.array(trainData).astype(np.float)
        y = np.array(trainingSet)[:,columns].astype(np.float)
        clf = QDA()
        clf.fit(X, y)
        testData = np.array(testSet)[:,0:np.array(trainingSet).shape[1] - 1]
        X_test = np.array(testData).astype(np.float)
        y_test = np.array(testSet)[:,columns].astype(np.float)
        accuracy = clf.score(X_test,y_test)
        accuracy *= 100
        print("Accuracy %:",accuracy)
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )


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