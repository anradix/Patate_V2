import os
import numpy as np
import pandas as pd
import random

from keras.preprocessing.image import load_img, img_to_array

directory = 'datas'
experiment_name = "vivatech"

# Init vars
input_size = (96, 160, 3)
images = []
labels_speed = []
labels_dir = []

# Load data
dir_list = os.listdir(directory)

# Mix up data
random.shuffle(dir_list)

# Create shapes
for name in dir_list:
    filename = directory + '/' + name
    image = load_img(filename, target_size=input_size[:2])
    image = img_to_array(image)
    value_dir = float(name.split('_')[1])
    value_speed = float(name.split('_')[0])
    labels_dir.append(value_dir)
    labels_speed.append(value_speed)
    images.append(image)

# Normalize data
input = np.array(images)
input /= 255.0
labels_speed = np.array(pd.get_dummies(labels_speed))
labels_dir = np.array(pd.get_dummies(labels_dir))

ouput = []
ouput.append(labels_speed)
ouput.append(labels_dir)

# Build model
from models.old_patate import getOldModel

model = getOldModel()

model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

# Callbacks setup
from keras.callbacks import ModelCheckpoint, TensorBoard

h5_filename = "model_{}.h5".format(experiment_name)
best_checkpoint = ModelCheckpoint(
    h5_filename, monitor='val_loss',
    verbose=1, save_best_only=True, mode='min'
)
tbd = TensorBoard(
    log_dir='graph', histogram_freq=0,
    write_graph=True, write_images=True
)

h = model.fit(input, ouput, batch_size=64, epochs=1, validation_split=0.2, verbose=1, callbacks=[best_checkpoint, tbd])
