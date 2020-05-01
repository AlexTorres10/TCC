#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:59:30 2020

@author: alextorres10

Mesmo da Ativ3, só que com a database inteira,
feita no train_test_split e com 60%-40%
"""


import pandas as pd
#import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("Vamos ler!")
df = pd.read_csv("allfotos.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=42)

print("Treino e teste prontos")

# SVC
svc = SVC(kernel='linear',gamma='auto',random_state=42)
svc.fit(X_train,y_train)
pred_svc = svc.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=3,p=1)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)

# NC
nc = NearestCentroid()
nc.fit(X_train,y_train)
pred_nc = nc.predict(X_test)


print("Acurácia do SVC:",accuracy_score(y_test, pred_svc)*100,"%")
print("Acurácia do KNN:",accuracy_score(y_test, pred_knn)*100,"%")
print("Acurácia do NC:",accuracy_score(y_test, pred_nc)*100,"%")