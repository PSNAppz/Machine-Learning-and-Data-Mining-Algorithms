import numpy as np
from sklearn.cluster import KMeans
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import os
import sys
from dotenv import load_dotenv

# environment variable
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET')

def call_function():
	# prepare data
    try:
        trainingSet = []
        loadDataset("/".join([DATASET_FOLDER, 'dnd/MACCS166.csv']),trainingSet)
        dataset=trainingSet
        X = np.array(dataset)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        print(kmeans.labels_)
        print(kmeans.cluster_centers_)
        y_kmeans = kmeans.predict(X)

        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )


def loadDataset(filename,trainingSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        for y in range(np.array(dataset).shape[1]):
	            dataset[x][y] = float(dataset[x][y])
	        trainingSet.append(dataset[x])
	

