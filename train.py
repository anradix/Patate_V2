# Basics
import os
import time
import numpy as np
import random

# Keras
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam

# Local
from models.old_patate import getOldModel
from models.dense_patate import getDenseModel

from utils.input_generator import InputGenerator
from utils.training import step_decay, PrintLR
from utils.argparser import get_args_training

# Init vars
experiment, nbrepoch, batchsize, validation, training = get_args_training()

experiment = experiment+"_"+time.strftime("%Y%m%d%H%M%S")
input_size = (96, 160, 3)

# Input generator
i_gen_train = InputGenerator(datapath=training, input_size=input_size)
i_gen_test = InputGenerator(datapath=validation, input_size=input_size)

nb_steps_train = i_gen_train.size // batchsize
nb_steps_test = i_gen_test.size // batchsize

# Build model
model = getOldModel(input_size=input_size)

adam = Adam(lr=10e-3)

model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

# Callbacks setup

directory = "graph/" + experiment
os.mkdir(directory)
h5_filename = "{}/model.h5".format(directory)

best_checkpoint = ModelCheckpoint(
    h5_filename, monitor='val_loss',
    verbose=1, save_best_only=True, mode='min'
)
tbd = TensorBoard(
    log_dir=directory, histogram_freq=0,
    write_graph=True, write_images=True
)
lrs = LearningRateScheduler(step_decay)
lrp = PrintLR()

h = model.fit_generator(
    i_gen_train.generator(batchsize),
    steps_per_epoch=nb_steps_train,
    epochs=nbrepoch,
    verbose=1,
    callbacks=[best_checkpoint, tbd, lrs, lrp],
    validation_data=i_gen_test.generator(batchsize),
    validation_steps=nb_steps_test,
    max_queue_size=10, shuffle=True, initial_epoch=0,
)
