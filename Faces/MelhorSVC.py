#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:18:17 2020

@author: alextorres10
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("Vamos ler!")
df = pd.read_csv("allfotos.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=42)

krnl = ['linear', 'rbf', 'sigmoid']
gm = ['auto', 'scale']
for i in krnl:
    for j in gm:
        svc = SVC(kernel=i,gamma=j,random_state=42)
        svc.fit(X_train,y_train)
        pred_svc = svc.predict(X_test)
        print("Kernel:",i,"Gamma:",j)
        print("Acurácia:",accuracy_score(y_test, pred_svc))