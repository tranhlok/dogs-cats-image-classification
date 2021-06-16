# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:34:40 2021

@author: tran_l8
"""

from keras.preprocessing import image
import os
import pathlib
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

datagen = ImageDataGenerator()

cur = pathlib.Path("Project3.py").parent.absolute()

dat = os.path.join(cur, "train")

train_dir = os.path.join(cur, "TrainingSet")
train_cat = os.path.join(train_dir, "Cat")

fnames = [os.path.join(train_cat, fname) for fname in os.listdir(train_cat)]

img_path = fnames[fnames.index(os.path.join(train_cat, 'cat.643.jpg'))]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    name = "cat"+str(i)+".png"
    try:
        plt.savefig(name)
    except:
        pass
    i += 1
    if i % 4 == 0:
        break

plt.show()
