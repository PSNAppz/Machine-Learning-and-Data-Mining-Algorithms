import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from numpy import loadtxt

# Load the iris dataset
dataset = loadtxt('../Dataset/bcancer.data', delimiter=",")
# split data into X and y
X = dataset[:,0:np.array(dataset).shape[1] - 1]
y = dataset[:,np.array(dataset).shape[1] - 1]
# Create a list of feature names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)

# Print the gini importance of each feature
print("Gini Importance of each feature")
print("-------------------------------")
for feature in clf.feature_importances_:
    print(feature)
print("-------------------------------")

# Create a selector object that will use the random forest classifier to identify
sfm = SelectFromModel(clf, threshold=0.01)

# Train the selector
sfm.fit(X_train, y_train)


# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature Model
print("Before Feature Selection",accuracy_score(y_test, y_pred))

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature  Model
print("After Feature Selection",accuracy_score(y_test, y_important_pred))