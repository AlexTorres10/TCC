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
from sklearn.decomposition import PCA
from Valida_Rec_faces_hc_PCA import validador
#num_pastas=int(input('entre com o numero de pastas para compor a base:'))
#num_imag=int(input('entre com o numero de imagens de cada pasta:'))
dB_Img0=[]
media=[]
pstx0 ='att_faces/s' 
##                     carrega uma umagem qualquer para pega a altura(H) e a largura{L} ##########
pstx2 =str(pstx0)+ str(1)
pstx1='/'+ str(1)
pstx3=str(pstx1)+'.jpg' 
pstx4=str(pstx2) + str(pstx3)
image = cv2.imread(str(pstx4))
imgtst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
H=int(imgtst.shape[0])
L=int(imgtst.shape[1])


num_imag=4   ##  No. de imagens de cada pasta para compr a base  ###

num_pastas=6 ## No. de pastas para compor a base ###

## estrutur de atrizes e vetors que serão utilizdos como varáveis ###        
Mat=np.zeros((H,L), dtype=np.uint8)
Matt=np.ones((num_imag*num_pastas,H*L), dtype=np.float)
COV=np.zeros((num_imag,num_imag), dtype=np.float)
In=np.zeros((H*L,1), dtype=np.float)
Inout=np.uint8(np.zeros((H*L,1)),dtype=np.uint8)
InBase=np.uint8(np.zeros((H*L,1)))
Inbasex=np.uint8(np.zeros((H*L,1)),dtype=np.uint)
Inbasex2=np.uint8(np.zeros((H*L,1)))
#A=np.zeros((H*L,num_imag),dtype=cv2.uint8)
#B=np.zeros((num_imag,H*L),dtype=np.float)
#Matp=np.ones((H*L+1,num_imag*num_pastas), dtype=np.float)
#data_rotulo=np.zeros((num_imag+1),dtype=np.float)1
R=0
Vet_att=[]
ind=[]

############  Gera a Base em forma de matrix ###
for cni in range(1,num_pastas+1):
        for cnj in range(0,num_imag):  
            pstx0 ='att_faces/s' 
            pstx2 =str(pstx0)+ str(cni)
            pstx1='/'+ str(cnj+1)
            pstx3=str(pstx1)+'.jpg' 
            pstx4=str(pstx2) + str(pstx3)
            image = cv2.imread(pstx4)
            imgtstx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       #     imgtst= cv2.Canny(imgtstx, 100,100)
          #  cv2.imshow("imagem mais proxima", imgtst)
      #      cv2.waitKey(0) 
          #  cv2.imshow('',image)
            print(pstx4)
            #cv2.imshow("Nome da janela", imgtst) 
            #cv2.waitKey(2)
            In=np.reshape(imgtstx, (H*L,1))
            for i in range(0,H*L):
               Matt[R][i]=In[i]          
            R=R+1     
           # X=np.reshape(canny2, (H*L,1))

########### define a estrutura do vetor de classificação
kclass=np.zeros((num_imag*num_pastas,1),dtype=np.float)


### Transforma a matriz em um data frame 
data_table0=pd.DataFrame(data=Matt) 

###   classifica o datafrae em classes através de um clsuter hmano ####
## cada pasta é uma classe  ### 
u=0
for i in range(0,num_pastas*num_imag):
        if((i % num_imag)==0):
            u=u+1
        kclass[i]=u

### junta a classe aos atirbutos do dataframe ###
data_table0['kclasse']=kclass
#data_table0['kclasse']=kmeans.labels_   
Out=[]
counter=0
Out=data_table0['kclasse']
       
#### aplica os algoritmos clássicos de IA
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn import tree  
from sklearn.svm import SVC
 
y = data_table0['kclasse']
x = data_table0.drop(['kclasse'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train, y_train)
knn2= KNeighborsClassifier(n_neighbors=2)
knn2.fit(x_train, y_train)


# Teste
#previsaoKNN = knn.predict(x_test)
#print(previsaoKNN)

nimg=0
Z=[]
nimg=0
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
Rmin_in=[]
R1=[]
R2=[]
R3=[]
R4=[]
R5=[]
R6=[]
R7=[]
#while(1 < 1000):
#        nps=int(input('entre com a pasta para reconhecimento:  '))
#        ni=int(input('entre com a imgem para reconhecimento:  '))
#KXR=[]
res_tot=[]
########### entra com as imagens a serem classsificadas
for nps in range(1,9):
    for ni in range(1,9): 
        pstx0 ='att_faces/s' 
        pstx2 =str(nps)
        pstx3= '/'
        pstx4=str(pstx0) + str(pstx2)+str(pstx3)
        pstx1=str(ni)
        pstx5='.jpg'
        pstx6x=str(pstx4)+str(pstx1)+str(pstx5)  
        
        print(pstx6x)
        imagex= cv2.imread(str(pstx6x))
        imgtstx0 = cv2.cvtColor(imagex,cv2.COLOR_BGR2GRAY)
      #  imgtstx= cv2.Canny(imgtstx0,100,100)
#        cv2.imshow("imagem teste", imgtstx) 
#        cv2.waitKey(0)
####### aqui fazemos uma leve rotação e trnslação da imagem a ser pesquisada                         
      
        centro = (L// 2, H // 2) #acha o centro
        M1 = cv2.getRotationMatrix2D(centro, 10, 1.0) #30 graus
        img_rotacionadop = cv2.warpAffine(imgtstx0, M1, (L,H))   
        M2 = cv2.getRotationMatrix2D(centro, -10, 1.0) #30 graus
        img_rotacionadon = cv2.warpAffine(imgtstx0, M2, (L,H))
      #  cv2.resize(img,img,92,112)
        
        deslocamentod = np.float32([[1, 0, 5], [0, 1, 5]])
        img_deslocadod = cv2.warpAffine(imgtstx0, deslocamentod, (L,H))
        
        
        deslocamentoe = np.float32([[1, 0, -5], [0, 1, 0]])
        img_deslocadoe= cv2.warpAffine(imgtstx0, deslocamentoe, (L,H))
        
        deslocamentoup = np.float32([[1, 0, 0], [0, 1, 5]])
        img_deslocadoup = cv2.warpAffine(imgtstx0, deslocamentoup, (L,H))
        
        deslocamentodown = np.float32([[1, 0, 0], [0, 1, -5]])
        img_deslocadodown = cv2.warpAffine(imgtstx0, deslocamentodown, (L,H))
        
        #### faz a classificação das imagens e suas rotações e translações  ########
        Inx=np.reshape(imgtstx0, ((1,H*L)))
        KX1=int(knn1.predict(Inx)) 
        
        Inx0=np.reshape(img_rotacionadop, ((1,H*L)))
        KX2=int(knn1.predict(Inx0))
        
        Inx1=np.reshape(img_rotacionadon, ((1,H*L)))
        Inx2=Inx1
        KX3=int(knn1.predict(Inx1))
        
        Inx20=np.reshape(img_deslocadod, ((1,H*L)))
        Inx2=Inx20
        KX4=int(knn1.predict(Inx20))
        
        Inx3=np.reshape(img_deslocadoe, ((1,H*L)))
        Inx2=Inx3
        KX5=int(knn2.predict(Inx3))
        
        Inx4=np.reshape(img_deslocadodown, ((1,H*L)))
        Inx2=Inx4
        KX6=int(knn2.predict(Inx4))
        

### As classificações sao colocadas em um vetor KXR ####
        countkx=0
        KXR=[KX1,KX2,KX3,KX4,KX5,KX6]
        data_KXR=pd.DataFrame(data=KXR) # calcula a moda 
        KXdf = data_KXR.mode()"# resultado da moda
        print(len(KXdf))
        for k in range(0,len(KXR)):
           if(KXR[k]== KXdf[0][0]):  
               countkx=countkx+1
        if  (countkx >= 3):
            KX=KXdf[0][0]   ## valor da classificação  #####
      
        #### exibe a imagem encontrada na pasta KX  #######
        print(KXR)
        print(KX)
        print ('Os K ',data_KXR)
        pstx0 ='att_faces/s' 
        pstx2 =str(pstx0)+ str(KX)
        pstx3= '/' + str(1)
        pstx4=str(pstx2) + str(pstx3)
        pstx5='.jpg'
        pstx6=str(pstx4)+ str(pstx5)                                                                                       
        Imagx=cv2.imread(pstx6)
        Imag0=cv2.cvtColor(Imagx, cv2.COLOR_BGR2GRAY)
 #       imshow