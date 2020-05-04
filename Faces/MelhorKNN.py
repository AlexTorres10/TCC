#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:27:56 2020

@author: alextorres10
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Vamos ler!")
df = pd.read_csv("allfotos.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=42)

n = [3, 5, 7]
p = [1, 2]
weight = ['uniform','distance']
for i in n:
    for j in p:
        for k in weight:
            knn = KNeighborsClassifier(n_neighbors=i,p=j,weights=k)
            knn.fit(X_train,y_train)
            pred_knn = knn.predict(X_test)
            print("K-vizinhos:",i,"P:",j,"Weight:",k)
            print("Acurácia:",accuracy_score(y_test, pred_knn))