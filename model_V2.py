from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers import concatenate
import keras.optimizers as Optimizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

def raw_img(img, name, filename, images, labels_speed, labels_dir):
    img = img_to_array(img)
    value_dir = float(name.split('_')[1])
    value_speed = float(name.split('_')[0])
    labels_dir.append(value_dir)
    labels_speed.append(value_speed)
    images.append(img)

def gussian_noise(img, name, filename, images, labels_speed, labels_dir):
    seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.Crop(percent=(0, 0.2))])
    
    #np.array(img)
    #images_aug = seq.augment_images(img)
    #print("Augmented:")
    #ia.imshow(np.hstack(images_aug))

def load_data(directory):
    images = []
    labels_speed = []
    labels_dir = []
    dir_list = listdir(directory)
    random.shuffle(dir_list)
    for name in dir_list:
        filename = directory + '/' + name
        img = load_img(filename, target_size=(96, 160))
        raw_img(img, name, filename, images, labels_speed, labels_dir)
        gussian_noise(img, name, filename, images, labels_speed, labels_dir)
    return images, labels_speed, labels_dir

directory = 'datas'
images, labels_speed, labels_dir = load_data(directory)

images = np.array(images)
images /= 255.0
labels_speed = np.array(pd.get_dummies(labels_speed))
labels_dir = np.array(pd.get_dummies(labels_dir))
print(images.shape)