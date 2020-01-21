import cv2
import numpy as np
import pandas as pd


df1 = np.uint8(np.array([range(30913)]))
df2 = np.uint8(np.array([range(30913)]))
npastas = 40
nfotos = 10
for i in range(1,npastas+1):
    s1 = "att_faces/s"
    s2 =(str(i)+"/")
    for j in range(1,nfotos+1):
        s3 = str(j)+".pgm"
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
'''
arr = np.asarray(dataset[1])
arr2 = arr.reshape(112,92,3)

cv2.imshow('Teste',arr2)
cv2.waitKey(5000)
cv2.destroyAllWindows()
'''