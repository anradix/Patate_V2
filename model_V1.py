from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers import concatenate
import keras.backend as K
import keras.optimizers as Optimizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import random

# LOAD DATA #

images = []
labels_speed = []
labels_dir = []
directory = 'datas'
dir_list = listdir(directory)
random.shuffle(dir_list)
for name in dir_list:
    filename = directory + '/' + name
    image = load_img(filename, target_size=(96, 160))
    image = img_to_array(image)
    value_dir = float(name.split('_')[1])
    value_speed = float(name.split('_')[0])
    labels_dir.append(value_dir)
    labels_speed.append(value_speed)
    images.append(image)

# PREPARE DATA #

images = np.array(images)
images /= 255.0
labels_speed = np.array(pd.get_dummies(labels_speed))
labels_dir = np.array(pd.get_dummies(labels_dir))

# CREATE MODEL #

img_in = Input(shape=(96, 160, 3), name='img_in')
x = img_in

x = Convolution2D(4, (5,5), strides=(2,2), use_bias=False)(x)       
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Convolution2D(8, (5,5), strides=(2,2), use_bias=False)(x)       
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Convolution2D(16, (5,5), strides=(2,2), use_bias=False)(x)       
x = BatchNormalization()(x)
x = Activation("relu")(x)


x = Flatten(name='flattened')(x)

x = Dense(100, use_bias=False)(x) 
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(.4)(x)
x = Dense(100, use_bias=False)(x)  
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(.4)(x)
x = Dense(100, use_bias=False)(x) 
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(.4)(x)

out_speed = Dense(2, activation='softmax')(x)
out_dir = Dense(5, activation='softmax')(x)

model = Model(inputs=[img_in], outputs=[out_speed, out_dir])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()

model_name="model_vivatech.h5"

best_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

labels=[]
labels.append(labels_speed)
labels.append(labels_dir)

tbd = TensorBoard(
    log_dir='graph', histogram_freq=0,
    write_graph=True, write_images=True
)

h = model.fit(images, labels, batch_size=64, epochs=1, validation_split=0.2, verbose=1, callbacks=[best_checkpoint, tbd])
