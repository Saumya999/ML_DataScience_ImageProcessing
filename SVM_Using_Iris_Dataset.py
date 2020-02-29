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
import seaborn as sns 

########### Loading the Data From the URL and Load it through csv Reader Assigning the column name 

###################################################################################################

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']


data = pd.read_csv(url, names=names)

print(data.head())


##### Pairwise Feature Plot to observe 
sns.pairplot(data,hue='species',palette='Set2')

######################## Train -Test Split #######################

##########  Data and Class Making #################
X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


#######################  Making the Model  ######################

from sklearn.svm import SVC

model= SVC(kernel='rbf')

################# fitting the Model ##########################

model.fit(X_train,Y_train)

############# PRedicting the Result ##########################

Y_Pred = model.predict(X_test)

######################## Checking the accuracy #################

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(accuracy_score(Y_test,Y_Pred))
print(confusion_matrix(Y_test,Y_Pred))
print(classification_report(Y_test,Y_Pred))











