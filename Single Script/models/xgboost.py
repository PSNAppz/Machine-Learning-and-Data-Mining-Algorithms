from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
from dotenv import load_dotenv

# environment variable
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET')


def call_function():
    try:
        # load data
        dataset = loadtxt("/".join([DATASET_FOLDER, 'iris.data']), delimiter=",")
        # split data into X and y
        X = dataset[:,0:np.array(dataset).shape[1] - 1]
        Y = dataset[:,np.array(dataset).shape[1] - 1]
        # split data into train and test sets
        seed = 1
        test_size = 0.25
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        # fit model no training data
        model = XGBClassifier()
        model.fit(X_train, y_train)
        # make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )    
