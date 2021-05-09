import numpy as np 
import pyswarms as ps 
import csv
import random
from numpy import genfromtxt
from sklearn import linear_model
np.set_printoptions(threshold=np.nan)
import os
import sys
from dotenv import load_dotenv


# environment variable
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET')
DATASET_OUTPUT_FOLDER = os.getenv('DATASET_OUTPUT')


# Load data from the given benchmarked datasets
def loadDataset(filename,testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(np.array(dataset).shape[1]):
                dataset[x][y] = float(dataset[x][y])
                testSet.append(dataset[x])
            


def call_function():
    try:
        #trainingSet=[]
        testSet=[]
        #split = 0.25
        loadDataset("/".join([DATASET_FOLDER, 'heart.data']),testSet)
        mtx = genfromtxt("/".join([DATASET_FOLDER, 'heart.data']), delimiter=',')
        trainData = np.array(testSet)[:,0:np.array(testSet).shape[1] - 1]
        columns = trainData.shape[1] 
        X = np.array(trainData).astype(np.float)
        y = np.array(testSet)[:,columns].astype(np.float)

        # Create an instance of the classifier
        classifier = linear_model.LogisticRegression()
        # Define objective function

         # Initialize swarm, arbitrary
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
        # Call instance of PSO
        dimensions = 13 # dimensions should be the number of features
        optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

        # Perform optimization
        cost, pos = optimizer.optimize(f(classifier=classifier, X=X, y=y), print_step=100, iters=1000, verbose=2)

        # Create two instances of LogisticRegression
        classifier = linear_model.LogisticRegression()

        # Get the selected features from the final positions
        X_selected_features = X[:,pos==1]  # subset

        # Perform classification and store performance in P
        classifier.fit(X_selected_features, y)

        # Compute performance
        subset_performance = (classifier.predict(X_selected_features) == y).mean()

        print('Subset performance: %.3f' % (subset_performance))
        Ycol = np.array(mtx)[:,dimensions]

        #X_selected_features = np.concatenate((X_selected_features,Ycol.reshape(Ycol.shape[0],1).astype(int)),axis=1)
        np.savetxt("/".join([DATASET_OUTPUT_FOLDER, 'PSOData.csv']),X_selected_features,fmt='%10.2f',delimiter=",")
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )


def f_per_particle(m, alpha, classifier, X, y):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """

    total_features = 13
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]    
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.88, classifier, X, y):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha, classifier, X, y) for i in range(n_particles)]
    return np.array(j)

   
