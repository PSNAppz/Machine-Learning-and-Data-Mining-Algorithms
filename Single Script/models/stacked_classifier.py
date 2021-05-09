import numpy as np 
import os
import sys
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

# environment variable
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET')

def call_function():
    try:
        dataset = np.loadtxt("/".join([DATASET_FOLDER, 'dnd/MACCS166.csv']), delimiter=",")
        # split data into X and y
        X = dataset[:,0:np.array(dataset).shape[1] - 1]
        y = dataset[:,np.array(dataset).shape[1] - 1]

        clf1 = LinearDiscriminantAnalysis()
        clf2 = RidgeClassifier()
        clf4 = RandomForestClassifier()
        clf3 = GaussianNB()
        sclf = StackingClassifier(classifiers=[clf1, clf3, clf4], 
                                meta_classifier=clf2)

        print('10-fold cross validation:\n')

        for clf, label in zip([clf1, clf2, clf4, sclf], 
                            ['LDA', 
                            'Gaussian Naive Bayes', 
                            'Random Forest',
                            'Meta - Ridge Classifier']):
            scores = model_selection.cross_val_score(clf,X,y, cv=10, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )
                                                 