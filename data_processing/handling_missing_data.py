#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 07:38:39 2017

@author: ashutosh
"""

import pandas as pd

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:,1:3] = imputer.fit_transform(X[:, 1:3])

#print data to see result
for x in X:
    print(x)