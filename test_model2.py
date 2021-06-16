# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:54:39 2021

@author: tran_l8
"""

import keras
import os, shutil, random
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pathlib
import matplotlib.pyplot as plt


model = load_model("cats_and_dogs_small_2.h5")

cur = pathlib.Path("test_model2.py").parent.absolute()
test_dir = os.path.join(cur, "TestingSet")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size = 20,
        class_mode='binary')


y = model.predict_generator(test_generator)
loss, acc = model.evaluate_generator(test_generator)
print(acc)
