

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:40:53 2019

@author: Saumyadipta Sarkar
@email: saumyadiptasarkar49@gmail.com
@phone: +918292310994
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

########### Loading the Data From the URL and Load it through csv Reader Assigning the column name 

###################################################################################################

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


data = pd.read_csv(url, names=names)

print(data.head())

print(data.describe(include='all'))
#####################################################################
##     Train and Test Split the Whole Data Set          #############
#####################################################################

X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


#####################################################################
##                   Feature Scaling          #######################
#####################################################################

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(X_train)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)


######################################################################
##             Classification Time Using KNN   #######################
######################################################################

from sklearn.neighbors import KNeighborsClassifier

## Defining the Classifier

classifier = KNeighborsClassifier(n_neighbors = 5,metric='euclidean')

classifier.fit(X_train,Y_train)


####### Prediction Time Over the created Model   #############

Y_Pred = classifier.predict(X_test)


########   Result Analysis  #########################

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

print(confusion_matrix(Y_test, Y_Pred))

print(classification_report(Y_test,Y_Pred))

print(accuracy_score(Y_test,Y_Pred))
########################################################################
#
#  ERROR Rate Measurement Over the Number of Cluster Definition 
#
########################################################################


error = []
for i in range (1,40):
    knn_classifier = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    knn_classifier.fit(X_train,Y_train)
    Y_Pred_i = knn_classifier.predict(X_test)
    error.append(np.mean(Y_Pred_i != Y_test))


###################################################################
############    PLotting the error Rate ###########################
###################################################################

plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color = 'red',linestyle='dashed',
         marker='o',markerfacecolor='blue',markersize=10)

plt.xlabel("K- values")
plt.ylabel("Mean Error")
plt.title("Average error vs K plot")





