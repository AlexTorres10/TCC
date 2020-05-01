#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:25:09 2020

@author: alextcarvalho
"""

import numpy as np
import cv2

pasta = 10
nfotos = 8
s1 = "Reduz/"
s2 =(str(pasta)+"/")
for j in range(1,nfotos+1):
    if (pasta < 10):
        s3 = "10"+str(pasta)+"_"
    else:
        s3 = "1"+str(pasta)+"_"
    s4 = str(j)+".tif"
    s = s1+s2+s3+s4
    print(s)
    img = cv2.imread(s)
    dedo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, dedo) = cv2.threshold(dedo, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('Dedo do cara original',img)
    cv2.imshow('Dedo do cara sem filtro',dedo)
    cv2.imwrite(s4, dedo)
    
    cv2.waitKey(50)
    cv2.destroyAllWindows()