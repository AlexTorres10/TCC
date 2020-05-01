import cv2
import numpy as np
import pandas as pd

comp = 128
larg = 128
dim = (comp, larg)
df1 = np.uint8(np.array([range(comp*larg*3+1)]))
df2 = np.uint8(np.array([range(comp*larg*3+1)]))
npastas = 5
nfotos = 20
qtdtrain = 0
qtdtest = 0
for i in range(npastas):
    s1 = "faces_pp40_bw/"
    s2 =(str(i)+"/")
    for j in range(nfotos):
        if ((i == 0 and j < 5) or i > 0):
            s3 = str(j)+".jpg"
            s = s1+s2+s3
        else:
            break
        print(s)
        img = cv2.imread(s)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        a = np.array(img)
        b = np.array([i])
        flat_arr = a.ravel()
        flat_arr = np.append(b,flat_arr,axis=None)
        vector = np.matrix(flat_arr)
        if ((i == 0 and j < 3) or (i > 0 and j < 10)):
            df1 = np.append(df1,vector,axis=1)
            qtdtrain+=1
        else:
            df2 = np.append(df2,vector,axis=1)
            qtdtest+=1
df1 = df1.reshape(qtdtrain+1,comp*larg*3+1)
df2 = df2.reshape(qtdtest+1,comp*larg*3+1)
print("Refeito o reshape")
np.savetxt("pp40_train.csv", df1, delimiter=",")
np.savetxt("pp40_test.csv", df2, delimiter=",")

print("Renomeando os datasets de treino")
df = pd.read_csv("pp40_train.csv")
df.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

df.to_csv (r'pp40_train.csv', index = None, header=True)
#Don't forget to add '.csv' at the end of the path

print("Renomeando os datasets de teste")
dft = pd.read_csv("pp40_test.csv")
dft.rename(columns={'0.000000000000000000e+00': 'out'
                   },inplace=True)

dft.to_csv (r'pp40_test.csv', index = None, header=True)
'''
arr = np.asarray(dataset[1])
arr2 = arr.reshape(112,92,3)

cv2.imshow('Teste',arr2)
cv2.waitKey(5000)
cv2.destroyAllWindows()
'''