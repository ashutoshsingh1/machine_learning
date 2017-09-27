#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 07:41:16 2017

@author: ashutosh
"""
import pandas as pd

data = pd.read_csv("Data.csv")

X = data.iloc[:,:3].values
y = data.iloc[:,3].values

#Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

#Feature Scaling is available for most of the library
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""