#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:11:54 2020

@author: alextcarvalho

Testando um modelo KNN
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("fotos.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

test = pd.read_csv("test.csv")

t_X = test.drop(["out"], axis=1)
t_y = test["out"]

print("Treino e teste prontos")
n = [3,5,7]
p = [1,2]

knn = KNeighborsClassifier(n_neighbors=3,p=1)
knn.fit(X,y)
pred = knn.predict(t_X)
right = 0
wrong = 0
for k in range(len(pred)):
    if (pred[k] == t_y[k]):
        right+=1
    else:
        wrong+=1
acc = accuracy_score(t_y,pred)
print("Tive",right,"acertos e",wrong,"erros com",acc*100,"% de acurácia")