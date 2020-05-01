#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:44:47 2020

@author: dciot-10
"""

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
import cv2
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
from os import listdir
from matplotlib import image
'''
loaded_images = list()
for i in range(5):
	# load image
    if (i == 0):
        for j in range(5):
            img_data = image.imread('faces_pp40/' + str(i) + '/' + str(j) + '.jpg')
            loaded_images.append(img_data)
    else:
        for j in range(20):
            img_data = image.imread('faces_pp40/' + str(i) + '/' + str(j) + '.jpg')
            loaded_images.append(img_data)
'''    

df = pd.read_csv("pp40_train.csv")
X = df.drop(["out"], axis=1)
y = df["out"]

test = pd.read_csv("pp40_test.csv")

t_X = test.drop(["out"], axis=1)
t_y = test["out"]

print("Treino e teste prontos")    
    
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
        '''
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
                print("Posição",k,"ainda errada: A pessoa era a",int(t_y[k]))
        '''
        
acc = right/len(pred_svc)*100
print("Tive",right,"acertos e",wrong,"erros com",acc,"% de acurácia")
