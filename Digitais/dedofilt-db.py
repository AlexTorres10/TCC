#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:29:30 2020

@author: alextcarvalho
"""

import cv2
import numpy as np
import pandas as pd

comp = 144
larg = 90
tam = (comp*larg*3)+1

df1 = np.uint8(np.array([range(tam)]))
df2 = np.uint8(np.array([range(tam)]))
npastas = 10
nfotos = 8
for i in range(1,npastas+1):
    s1 = "Filt/"
    s2 =(str(i)+"/")
    for j in range(1,nfotos+1):
        s3 = str(j)+".tif"
        s = s1+s2+s3
        print(s)
        img = cv2.imread(s)
        a = np.array(img)
        b = np.array([i])
        flat_arr = a.ravel()
        flat_arr = np.append(b,flat_arr,axis=None)
        vector = np.matrix(flat_arr)
        if (j % 2 == 0):
            df1 = np.append(df1,vector,axis=1)
        else:
            df2 = np.append(df2,vector,axis=1)
df1 = df1.reshape(int((npastas*nfotos/2)+1),tam)
df2 = df2.reshape(int((npastas*nfotos/2)+1),tam)

print("Refeito o reshape")
np.savetxt("filttrain.csv", df1, delimiter=",")
np.savetxt("filttest.csv", df2, delimiter=",")

print("Renomeando os datasets de treino")
df = pd.read_csv("filttrain.csv")

df.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

df.to_csv (r'filttrain.csv', index = None, header=True)
#Don't forget to add '.csv' at the end of the path

print("Renomeando os datasets de teste")
dft = pd.read_csv("filttest.csv")
dft.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

dft.to_csv (r'filttest.csv', index = None, header=True)