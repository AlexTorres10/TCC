#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:11:54 2020

@author: alextcarvalho

Testando o melhor KNN, e enfiando um NearestCentroid em quem tá errado
KNC
DecisionTree
SVM
"""

import pandas as pd
#import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score

df = pd.read_csv("fotos.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

test = pd.read_csv("test.csv")

t_X = test.drop(["out"], axis=1)
t_y = test["out"]

print("Treino e teste prontos")
n = [3,5,7]
p = [1,2]

svc = SVC(kernel='linear',gamma='auto',random_state=42)
svc.fit(X,y)
pred_svc = svc.predict(t_X)
        
right = 0
wrong = 0
for k in range(len(pred_svc)):
    if (pred_svc[k] == t_y[k]):
        right+=1
    else:
        wrong+=1
        knn = KNeighborsClassifier(n_neighbors=3,p=1)
        knn.fit(X,y)
        pred_knn = knn.predict(t_X)

        if (pred_knn[k] == t_y[k]):
            right+=1
            wrong-=1
            print("Posição",k,"ajeitada pelo KNN: É da pessoa",int(pred_knn[k]))
        else:
            nc = NearestCentroid()
            nc.fit(X, y)
            pred_nc = nc.predict(t_X)
            if (pred_nc[k] == t_y[k]):
                right+=1
                wrong-=1
                print("Posição",k,"ajeitada pelo NC: É da pessoa",int(pred_nc[k]))
            else:
                print("Posição",k,"ainda errada: A pessoa era a",int(t_y[k]),"e disse que era a",int(pred_nc[k]))
        
        
acc = right/len(pred_svc)*100
print("Tive",right,"acertos e",wrong,"erros com",acc,"% de acurácia")