#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 07:20:41 2017

@author: ashutosh
"""


import pandas as pd

data = pd.read_csv("Data.csv")

X = data.iloc[:,:3].values
y = data.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:, 1:3])


#Encoading categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x = LabelEncoder()
X[:,0] = label_encoder_x.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

#Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

#Feature Scaling 
"""
Scale all independent variable into same range so no variable dominate

Two types of feature scalling

1. Standardization 
Xstd = (X-Xmean)/standard_deviation_of_X

2. Normalization
Xnormal = (X-Xmin)/(max-min)
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

