# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:29:42 2020

@author: Saumyadipta Sarkar
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

import pandas as pd 

X = pd.read_csv('F:/ML_Python/DataSets/Titanic_Data/train.csv')
y = X.pop('Survived')


## Describing the Data set 
print(X.describe())

# Age has missing so impute The Age Values with Mean
X['Age'].fillna(X.Age.mean(), inplace = True)

## Random Forest Only With the Numeric variables

numeric_Vars = list(X.dtypes[X.dtypes != 'object'].index)

# Building the Model 
model = RandomForestRegressor(n_estimators = 100,
                              oob_score = True,
                              random_state=42)

# Fitting the Model With Numeric variables 
model.fit(X[numeric_Vars], y)

# Playing With categorical Variables 

Categorical_Vars = list(X.dtypes[X.dtypes == 'object'].index)

X[Categorical_Vars].describe()

# dropping the unnecessary Variables
X.drop (['Name','Ticket', 'PassengerId'],axis=1, inplace = True)


def clean_Cabin (x):
    try:
        return x[0]
    except TypeError:
        return 'None'
X['Cabin'] = X.Cabin.apply(clean_Cabin)

cat_vars = ['Sex', 'Cabin', 'Embarked']

for variable in cat_vars:
    X[variable].fillna ('Missing',inplace=True)
    dummies = pd.get_dummies(X[variable],prefix = variable)
    X = pd.concat([X,dummies],axis=1)
    X.drop([variable], axis=1, inplace=True)
 
model = RandomForestRegressor(n_estimators = 1000,
                              oob_score = True,
                              random_state=42,
                              n_jobs = -1,
                              max_features ='auto',
                              min_samples_leaf=5)

model.fit(X,y)

print (model.oob_score_)
print(roc_auc_score(y,model.oob_prediction_))


print(model.feature_importances_)

feature_importance = pd.Series(model.feature_importances_, index = X.columns)

feature_importance.sort_values()

feature_importance.plot(kind = 'barh')