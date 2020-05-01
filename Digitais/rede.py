#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:05:13 2020

@author: alextcarvalho
"""

import tensorflow as tf
print("Versão do TensorFlow:", tf.__version__)

import keras as K
print("Versão do Keras:", K.__version__)

import pandas as pd
# Imports
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("train.csv")
x_train = df.drop(["out"], axis=1)
y_train = df["out"]

test = pd.read_csv("test.csv")

x_test = test.drop(["out"], axis=1)
y_test = test["out"]

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


x_train = x_train.reshape(x_train.shape[0], 90, 144, 3)
x_test = x_test.reshape(x_test.shape[0], 90, 144, 3)

input_shape = (90, 144, 3)

x_train = x_train.astype('float32')
x_test = x_train.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
# Inicializando a Rede Neural Convolucional
model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

model.evaluate(x_test, y_test)

#image_index = 2
#plt.imshow(x_test[image_index].reshape(90, 144,3),cmap='Greys')
#pred = model.predict(x_test[image_index].reshape(1, 90, 144, 3))
#print(pred.argmax())
'''
# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), 
                      activation = 'relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory('/home/dciot-10/Área de Trabalho/TCC/Dedos/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('/home/dciot-10/Área de Trabalho/TCC/Dedos/Test',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 9000,
                         epochs = 10,
                         validation_data = validation_set,
                         validation_steps = 4000)
'''