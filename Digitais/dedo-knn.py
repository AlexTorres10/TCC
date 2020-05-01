#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:51:47 2020

@author: dciot-10
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("socofinger.csv")
X = df.drop(["out"], axis=1)
y = df["out"]
X_train, t_X, y_train, t_y = train_test_split(
    X, y, test_size=0.3, random_state=42)

t_y = t_y.to_numpy()
print("Treino e teste prontos")

'''
print("Lendo train")
df = pd.read_csv("soco_train.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

print("Lendo test")
test = pd.read_csv("soco_test.csv")

t_X = test.drop(["out"], axis=1)
t_y = test["out"]
'''
n = [3,5,7]
p = [1,2]

for i in n:
    for j in p:
        knn = KNeighborsClassifier(n_neighbors=i,p=j)
        knn.fit(X_train,y_train)
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