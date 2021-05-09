import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



COMPONENT_NUM = 2

def call_function(DATASET_FOLDER='Dataset', DATASET_OUTPUT_FOLDER='Dataset/output', DATA='Morgan.csv'):

    try:

        print('Read training data...')
        with open("/".join([DATASET_FOLDER, DATA]), 'r') as reader:
            reader.readline()
            train_label = []
            train_data = []
            for line in reader.readlines():
                data = list(map(float, line.rstrip().split(',')))
                train_label.append(data[0])
                train_data.append(data[1:])
        print('Loaded ' + str(len(train_label)))

        print('Reduction...')
        train_label = numpy.array(train_label)
        train_data = numpy.array(train_data)
        pca = PCA(n_components=COMPONENT_NUM, whiten=True)
        pca.fit(train_data)
        train_data = pca.transform(train_data)

        X_train,X_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.2,random_state=32)

        print('Train SVM...')
        svc = SVC(C=10.0,kernel="rbf",gamma=0.1)
        svc.fit(X_train, y_train)

        #print('Read testing data...')
        #with open('../input/test.csv', 'r') as reader:
        #   reader.readline()
        #  test_data = []
        # for line in reader.readlines():
            #    pixels = list(map(int, line.rstrip().split(',')))
            #   test_data.append(pixels)
        #print('Loaded ' + str(len(test_data)))

        print('Predicting...')
        predict = svc.predict(X_test)
        acc=accuracy_score(predict,y_test)
        print ("Accuracy % : ",acc)
        #print('Saving...')
        #with open('predict.csv', 'w') as writer:
        #   writer.write('"ImageId","Label"\n')
        #  count = 0
        # for p in predict:
            #    count += 1
        #     writer.write(str(count) + ',"' + str(p) + '"\n')#
    except Exception as e:
       
        print( "Error: %s" % e )