#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:32:36 2020

@author: alextorres10
"""


import os #biblioteca necessária para acessar os arquivos
import numpy as np
import pandas as pd
import cv2

addr = "./256_ObjectCategories"

for folder in os.listdir(addr): # para cada nome no diretório atual
    i = int(folder[0:3])
    fold = addr + "/" + folder
    for obj in os.listdir(fold):
        img_ad = fold + "/" + obj
        img = cv2.imread(img_ad)
        a = np.array(img)
        b = np.array([i])
        flat_arr = a.ravel()
        flat_arr = np.append(b,flat_arr,axis=None)
        vector = np.matrix(flat_arr)
        df1 = np.append(df1,vector,axis=1)
df1 = df1.reshape(int((npastas*nfotos/2)+1),30913)
df2 = df2.reshape(int((npastas*nfotos/2)+1),30913)
print("Refeito o reshape")
np.savetxt("fotos.csv", df1, delimiter=",")
np.savetxt("test.csv", df2, delimiter=",")

print("Renomeando os datasets de treino")
df = pd.read_csv("fotos.csv")
df.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

df.to_csv (r'fotos.csv', index = None, header=True)
#Don't forget to add '.csv' at the end of the path

print("Renomeando os datasets de teste")
dft = pd.read_csv("test.csv")
dft.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

dft.to_csv (r'test.csv', index = None, header=True)