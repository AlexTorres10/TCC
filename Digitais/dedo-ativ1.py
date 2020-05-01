import cv2
import numpy as np
import pandas as pd

comp = 96
larg = 103
tam = (comp*larg*3)+1

df1 = np.uint8(np.array([range(tam)]))
df2 = np.uint8(np.array([range(tam)]))
people = 40
hand = ["Left", "Right"]
fingers = ["indexfinger","littlefinger", "middlefinger"
           ,"ringfinger", "thumbfinger"]
ntr = 0
nte = 0
for i in range(people):
    s = "Dedos/SOCOFing/Real"
    for j in hand:
        for k in fingers:
            s = s + str(i+1) + j + k + ".BMP"
            img = cv2.imread(s)
            
            a = np.array(img)
            b = np.array([i])
            flat_arr = a.ravel()
            flat_arr = np.append(b,flat_arr,axis=None)
            vector = np.matrix(flat_arr)
            if (j % 2 == 0):
                df1 = np.append(df1,vector,axis=1)
                ntr+=1
            else:
                df2 = np.append(df2,vector,axis=1)
                nte+=1
df1 = df1.reshape(int((people*ntr)+1),tam)
df2 = df2.reshape(int((people*nte)+1),tam)

print("Refeito o reshape")
np.savetxt("india_train.csv", df1, delimiter=",")
np.savetxt("india_test.csv", df2, delimiter=",")

print("Renomeando os datasets de treino")
df = pd.read_csv("train.csv")

df.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

df.to_csv (r'train.csv', index = None, header=True)
#Don't forget to add '.csv' at the end of the path

print("Renomeando os datasets de teste")
dft = pd.read_csv("test.csv")
dft.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

dft.to_csv (r'test.csv', index = None, header=True)