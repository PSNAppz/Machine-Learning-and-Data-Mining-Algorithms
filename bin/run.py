from models import div
from models import (svm,\
                    random_forest, \
                    qda, \
                    pca_svm, \
                    naive_bayes, \
                    naive, \
                    lda_classifier, \
                    kmeans, \
                    decisiontree, \
                    stacking_scratch, \
                    rotation_forest_pca, \
                    rotation_forest_lda, \
                    knn, \
                    rfe, \
                    randomforest_fs, \
                    bpso, \
                    xgboost, \
                    gradient_boosting, \
                    stacked_classifier, \
                    ada_boosting, \
                    bagged_svm, \
                    bagged_qda, \
                    bagged_lda, \
                    bagged_knn
                )
                    
# Specify the dataset name below
DATA='sonar.data'

# calls SVM model
print("SVM model : ")
svm.call_function(DATA=DATA)

div.divider()

print("XGBoost model : ")
xgboost.call_function(DATA=DATA)

div.divider()

# calls random forest model
print("Random Forest model : ")
random_forest.call_function(DATA=DATA)

div.divider()

# calls random forest model
print("QDA model : ")
qda.call_function(DATA=DATA)

div.divider()

print("PCA & SVM model : ")
pca_svm.call_function(DATA=DATA)

div.divider()

print("Naive Bayes model : ")
naive_bayes.call_function(DATA=DATA)

div.divider()

# print("Naive model : ")
# naive.call_function(DATA=DATA)

div.divider()

print("LDA Classifier model : ")
lda_classifier.call_function(DATA=DATA)

div.divider()

# print("KMeans model : ")
# kmeans.call_function(DATA=DATA)

# div.divider()

print("Decision Tree model : ")
decisiontree.call_function(DATA=DATA)

div.divider()

print("Stacked model : ")
stacking_scratch.call_function(DATA=DATA)

div.divider()

print("Stacked model : ")
stacked_classifier.call_function(DATA=DATA)

div.divider()

print("Rotation Forest PCA model : ")
rotation_forest_pca.call_function(DATA=DATA)

div.divider()

print("Rotation Forest LDA model : ")
rotation_forest_lda.call_function(DATA=DATA)

div.divider()

print("KNN model : ")
knn.call_function(DATA=DATA)

div.divider()

print("Feature Selection RFE model : ")
rfe.call_function(DATA=DATA)

div.divider()

print("Feature Selection Random Forest model : ")
randomforest_fs.call_function(DATA=DATA)

div.divider()

print("Feature Selection BPSO model : ")
bpso.call_function(DATA=DATA)

div.divider()

print("Gradient Boosting model : ")
gradient_boosting.call_function(DATA=DATA)

div.divider()

print("AdaBoosting model : ")
ada_boosting.call_function(DATA=DATA)

div.divider()

print("Bagged SVM model : ")
bagged_svm.call_function(DATA=DATA)

div.divider()

print("Bagged QDA model : ")
bagged_qda.call_function(DATA=DATA)

div.divider()

print("Bagged LDA model : ")
bagged_lda.call_function(DATA=DATA)
div.divider()

print("Bagged KNN model : ")
bagged_knn.call_function(DATA=DATA)
