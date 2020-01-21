#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:11:54 2020

@author: alextcarvalho

Testando vários modelos KNN de uma vez
KNC
DecisionTree
SVM
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("fotos.csv")
X = df.drop(["out"], axis=1)
y = df["out"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30,random_state=42)

test = pd.read_csv("test.csv")

t_X = test.drop(["out"], axis=1)
t_y = test["out"]

print("Treino e teste prontos")
n = [3,5,7]
p = [1,2]
for i in n:
    for j in p:
        knn = KNeighborsClassifier(n_neighbors=i,p=j)
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
        print("Com",i,"vizinhos e p",j,"tive",right,"acertos e ",wrong,"erros")
        print("Acurácia foi de",acc*100,"%")