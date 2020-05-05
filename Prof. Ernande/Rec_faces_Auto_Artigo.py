# -*- coding: utf-8 -*-


"""
Created on Sun Apr  8 11:06:15 2018

@author: ernande
"""
#import time as t
#def PCA_Digitais_Finger_Rec(M,Xx,Xy)
#from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv2
#from Valida_Rec_faces_hc_PCA import validador
from numpy import linalg as LA
#from sklearn.decomposition import PCA
#from Valida_Rec_faces_hc_PCA import validador
import os,os.path
dB_Img0=[]
media=[]
pstx0 ='att_faces/s' 
pstx2 =str(pstx0)+ str(1)
pstx1='/'+ str(1)
pstx3=str(pstx1)+'.jpg' 
pstx4=str(pstx2) + str(pstx3)
image = cv2.imread(str(pstx4))
#print(image)
imgtst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#H=int(imgtst.shape[1])
#L=int(imgtst.shape[0])
H,L=imgtst.shape

path = os.listdir('att_faces/')
N=len(path)
num_pastas=N
num_imag=14
#num_imag=4
#num_pastas=2   #        print(pstx4)        
Mat=np.zeros((H,L), dtype=np.uint8)
Matt=np.zeros((num_imag*num_pastas,H*L), dtype=np.float)
#COV=np.zeros((num_imag,num_imag), dtype=np.float)
In=np.zeros((H*L,1), dtype=np.float)
refk=np.zeros((num_imag,H*L), dtype=np.float)
dif=np.zeros((num_imag,H*L), dtype=np.float)
img_out=np.zeros((H,L), dtype=np.float)
img_outf=np.zeros((H,L), dtype=np.float)
img_outfx=np.zeros((H,L), dtype=np.float)
Inx=np.zeros((1,H*L), dtype=np.float)
Inx2=np.zeros((1,H*L), dtype=np.float)
Media=np.zeros((1,H*L), dtype=np.float)
dist=np.zeros((1,H*L), dtype=np.float)
dist0=np.zeros((1,H*L), dtype=np.float)
distx=np.ones((1,H*L), dtype=np.float)
Inout=np.uint8(np.zeros((H*L,1)),dtype=np.uint8)
InBase=np.uint8(np.zeros((H*L,1)))
Inbasex=np.uint8(np.zeros((H*L,1)),dtype=np.uint)
Inbasex2=np.uint8DIR='att_faces/'
path = "/"
dir = os.listdir(path)
for file in dir:
     if (file == "img00.jpg"):
            os.remove(file)
#A=np.zeros((H*L,num_imag),dtype=cv2.uint8)
#B=np.zeros((num_imag,H*L),dtype=np.float)
#Matp=np.ones((H*L+1,num_imag*num_pastas), dtype=np.float)
#data_rotulo=np.zeros((num_imag+1),dtype=np.float)1
R=0
Vet_att=[]
ind=[]
resp=[]
Rmin2=[]
for cni in range(0,num_pastas):
#        pathf = os.listdir('att_faces/s'+str(cni))
#        num_imag=len(pathf)
        for cnj in range(0,num_imag):  
            pstx0 ='att_faces/s' 
            pstx2 =str(pstx0)+ str(cni+1)
            pstx1='/'+ str(cnj+1)
            pstx3=str(pstx1)+'.jpg' 
            pstx4=str(pstx2) + str(pstx3)
            image = cv2.imread(pstx4)
            imgtstp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#            tamanho_novo = (H,L)
#            imgtst = cv2.resize(imgtstp,tamanho_novo, interpolation = cv2.INTER_AREA)
         #   imgtst= cv2.Canny(imgtstu, 10,10)
     #       cv2.imshow("imagem mais proxima", imgtst)
      #      cv2.waitKey(0) 
          #  cv2.imshow('',image)
            print(pstx4)
            #cv2.imshow("Nome da janela", imgtst) 
            
            #cv2.waitKey(2)
            In=np.reshape(imgtstp, (H*L,1))
            for i in range(0,H*L):
               Matt[R][i]=In[i]          
            R=R+1     
           # X=np.reshape(canny2, (H*L,1))
kclass=np.zeros((num_imag*num_pastas,1),dtype=np.float)
#from sklearn.cluster import KMeans       
data_table0=pd.DataFrame(data=Matt) 
 
u=0
Rmin=[]
for i in range(0,num_pastas*num_imag):
        if((i % num_imag)==0):
            u=u+1
        kclass[i]=u
    
data_table0['kclasse']=kclass
#data_table0['kclasse']=kmeans.labels_   
Out=[]
counter=0
Out=data_table0['kclasse']
       

from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import tree  
#from sklearn.svm import SVC
 
# Teste
#previsaoKNN = knn.predict(x_test)
#print(previsaoKNN)
resp1=[]
resp2=[]
for nj in range(1,19):
#nimg=0
   
    Z=[]
    #nimg=0
    Z=[]
    #while (nimg < 1):
    count=0
    acertos=0
    countkx=0
    countKX2=0
    countKX3=0
    count2=0
    arquivo = open('log_rec_face.txt', 'w')
    arquivo.close()
   # Rmin=[]
    Rmax_in=[]
    Rmax_Rmin_in=[]
    RmindivRmax_in=[]
    Rmin_out=[]
    Rmax_out=[]
    Rmax_Rmin_out=[]
    RmindivRmax_out=[]
    #while(1 < 1000):
    #        nps=int(input('entre com a pasta para reconhecimento:  '))
    #        ni=int(input('entre com a imgem para reconhecimento:  '))
    res_tot=[]
#for nps in range(1,40):
#for nj in range(1,18): 
    pstx0 ='att_facesX' 
    pstx2 =str(pstx0)
    pstx3= '/'
    pstx4=str(pstx2) + str(pstx3)
    pstx1=str(nj)
    pstx5='.jpg'
    pstx6x=str(pstx4)+ str(pstx1) +str(pstx5)  
    largura_nova=L
    altura_nova=H
    print(pstx6x)
    imagex= cv2.imread(pstx6x)
    imgtstx0v = cv2.cvtColor(imagex, cv2.COLOR_BGR2GRAY)
    tamanho_novo = (L,H)
    imgtstx0 = cv2.resize(imgtstx0v,tamanho_novo, interpolation = cv2.INTER_AREA)
#    cv2.imshow('in',imgtstx0)
    #cv2.waitKey(0)
    #imgtstx0= cv2.Canny(imgtstxv,10,10)
    #cv2.imshow("imagem teste", imgtstx0) 
   
    #  
    
    y = data_table0['kclasse']
    x = data_table0.drop(['kclasse'], axis=1)
    Inx=np.reshape(imgtstx0, ((1,H*L)))
    KXL=[]
    for li in range(0,20):
    
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        
        knn10 = KNeighborsClassifier(n_neighbors=1)
        knn10.fit(x_train, y_train)
        knn30 = KNeighborsClassifier(n_neighbors=2)
        knn30.fit(x_train, y_train)
        knn50 = KNeighborsClassifier(n_neighbors=3)
        knn50.fit(x_train, y_train)
    #    classifD3 = tree.DecisionTreeClassifier()
    #    classifD3.fit(x_train,y_train.values.ravel())
    #    clfSVM=SVC(kernel='rbf', C = 10.0, gamma=0.1)
    #    clfSVM.fit(x_train,y_train.values.ravel())
        
    
    
    #nc.fit(x_train, y_train)
    # 
    #
    #classifD3 = tree.DecisionTreeClassifier()
    #classifD3.fit(x_train,y_train.values.ravel())
    #clfSVM = SVC(kernel='rbf', C = 10.0, gamma=0.1)
    #clfSVM.fit(x_train,y_train.values.ravel())
        KX10=int(knn10.predict(Inx) )
        KX30=int(knn30.predict(Inx))
        KX50=int(knn50.predict(Inx))
    #    KXD=int(classifD3.predict(Inx))
    #    KXS=int(clfSVM.predict(Inx))
        KXL.append(KX10)
        KXL.append(KX30)
        KXL.append(KX50)
    #    KXL.append(KXD)
    #    KXL.append(KXS)  
    
    #KX4=int(classifD3.predict(Inx))
    #KX5=int(clfSVM.predict(Inx))
    
    countkx=0
    KXR=KXL
    countkx=0
    data_KXR=pd.DataFrame(data=KXR) 
    KXdf = data_KXR.mode()
    if(len(KXdf)>1):
        KX=KX10
    else:
        KX=KXdf[0][0]
    
    for j in range(1,num_imag):
        pstx0 ='att_faces/s' 
        pstx2 =str(pstx0) + str(KX)
        pstx3= '/'
        pstx4=str(pstx2) + str(pstx3)
        pstx5=str(j)+'.jpg'
        pstx6=str(pstx4)+ str(pstx5)   
        img_out0=cv2.imread(pstx6)
        img_outu = cv2.cvtColor(img_out0,cv2.COLOR_BGR2GRAY)
        tamanho_novo = (L,H)
        img_out = cv2.resize(img_outu,tamanho_novo, interpolation = cv2.INTER_AREA)
        distx=img_out
        
        refk[0:H*L][j]=np.reshape(distx, ((1,H*L)))
        for jk in range(0,num_imag):
            dif[jk][0:H*L]=abs(refk[jk][0:H*L]-Inx)            
    resp=int(min(LA.norm(dif,axis=0)))
    respx=(int(min(np.mean(dif,axis=0))))
    print(resp,respx)
    if (resp <=61) & (respx<=40):
#        cv2.imwrite('img00.jpg',img_outxe)  
#     #   cv2.imwrite('img00.jpg',img_out0)  
        msg=0
        arquivo=open('status_faces.txt', 'w')  
        arquivo.write(str(msg)+'\n')
        arquivo.close()  
        print('Na BASE')
    else:
        msg=1
        print('Fora da BASE')
        arquivo=open('status_faces.txt', 'w')   
        arquivo.write(str(msg)+'\n')
        arquivo.close()     
       # tm.sleep(10)
    
#    cv2.imshow('out',img_outxe)
#    
#    cv2.waitKey(0)
    Rmin.append(resp)
    Rmin2.append(respx)
        
plt.plot(Rmin2, linestyle='-.', color='blue', label="RMax/Rmax")
#plt.plot(Rmin2, linestyle='-.', color='black', label="RMax/Rmax")
plt.legend(loc='upper center')
plt.grid(True)
plt.xlabel('separador')
plt.ylabel('n')
plt.show()