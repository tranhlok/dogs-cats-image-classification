# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:17:34 2021

@author: tran_l8
"""

import keras
import os, shutil, random
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import pathlib
import matplotlib.pyplot as plt

def setup_data():
    cur = pathlib.Path("Project3.py").parent.absolute()

    dat = os.path.join(cur, "train")

    train_dir = os.path.join(cur, "TrainingSet")
    try:
        shutil.rmtree(train_dir)
    except:
        os.makedirs(train_dir)
    test_dir = os.path.join(cur, "TestingSet")
    try:
        shutil.rmtree(test_dir)
    except:
        os.makedirs(test_dir)
    val_dir = os.path.join(cur, "ValidationSet")
    try:
        shutil.rmtree(val_dir)
    except:
        os.makedirs(val_dir)

    train_cat = os.path.join(train_dir, "Cat")
    os.makedirs(train_cat)
    train_dog = os.path.join(train_dir, "Dog")
    os.makedirs(train_dog)

    test_cat = os.path.join(test_dir, "Cat")
    os.makedirs(test_cat)
    test_dog = os.path.join(test_dir, "Dog")
    os.makedirs(test_dog)

    val_cat = os.path.join(val_dir, "Cat")
    os.makedirs(val_cat)
    val_dog = os.path.join(val_dir, "Dog")
    os.makedirs(val_dog)

    train_im_dog = random.sample(range(0, 5001), 1000)
    test_im_dog = []
    val_im_dog = []
    while len(test_im_dog) < 500:
        n = random.randint(0, 5000)
        if n not in train_im_dog and n not in test_im_dog:
            test_im_dog.append(n)
    while len(val_im_dog) < 500:
        n = random.randint(0, 5000)
        if n not in train_im_dog and n not in test_im_dog and n not in val_im_dog:
            val_im_dog.append(n)

    train_im_cat = random.sample(range(0, 5001), 1000)
    test_im_cat = []
    val_im_cat = []
    while len(test_im_cat) < 500:
        n = random.randint(0, 5000)
        if n not in train_im_cat and n not in test_im_cat:
            test_im_cat.append(n)
    while len(val_im_cat) < 500:
        n = random.randint(0, 5000)
        if n not in train_im_cat and n not in test_im_cat and n not in val_im_cat:
            val_im_cat.append(n)

    dog_names = ["dog.{}.jpg".format(i) for i in train_im_dog]
    for _ in dog_names:
        src = os.path.join(dat, _)
        dst = os.path.join(train_dog, _)
        shutil.copy(src, dst)

    dog_names = ["dog.{}.jpg".format(i) for i in test_im_dog]
    for _ in dog_names:
        src = os.path.join(dat, _)
        dst = os.path.join(test_dog, _)
        shutil.copy(src, dst)

    dog_names = ["dog.{}.jpg".format(i) for i in val_im_dog]
    for _ in dog_names:
        src = os.path.join(dat, _)
        dst = os.path.join(val_dog, _)
        shutil.copy(src, dst)

    cat_names = ["cat.{}.jpg".format(i) for i in train_im_cat]
    for _ in cat_names:
        src = os.path.join(dat, _)
        dst = os.path.join(train_cat, _)
        shutil.copy(src, dst)

    cat_names = ["cat.{}.jpg".format(i) for i in test_im_cat]
    for _ in cat_names:
        src = os.path.join(dat, _)
        dst = os.path.join(test_cat, _)
        shutil.copy(src, dst)

    cat_names = ["cat.{}.jpg".format(i) for i in val_im_cat]
    for _ in cat_names:
        src = os.path.join(dat, _)
        dst = os.path.join(val_cat, _)
        shutil.copy(src, dst)
    return train_dir, val_dir

train_dir, val_dir = setup_data()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

model.save('cats_and_dogs_small_3.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
