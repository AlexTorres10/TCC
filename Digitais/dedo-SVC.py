#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:51:47 2020

@author: dciot-10
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("soco_train.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

test = pd.read_csv("soco_test.csv")

t_X = test.drop(["out"], axis=1)
t_y = test["out"]

print("Treino e teste prontos")
C = [1.0,2.0,3.0]
gam = ['scale','auto']
ker = ['rbf','sigmoid','precomputed']


for i in ker:
    for j in gam:
        for k in C:
            svc = SVC(C=k,kernel=i,gamma=j,random_state=42)
            svc.fit(X,y)
            pred = svc.predict(t_X)
            right = 0
            wrong = 0
            for l in range(len(pred)):
                if (pred[l] == t_y[l]):
                    right+=1
                else:
                    wrong+=1
            acc = accuracy_score(t_y,pred)
            print("Kernel:",i,"; Gamma:",j,"C:",k)
            print("Acertos:",right,"; Erros: ",wrong)
            print("Acurácia:",acc*100,"%")
