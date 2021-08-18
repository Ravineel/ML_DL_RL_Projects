#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

path = './DataSets/happy-or-sad.zip'
zip = zipfile.ZipFile(path,'r')
zip.extractall("./DataSets/h-s")
zip.close()
#%%
class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs= {}):
        if logs['acc'] >0.99:
            self.model.stop_training = True


callback = MyCallBack()

model = Sequential([
    Conv2D(64, 3, activation='relu', input_shape=(150,150,3)),
    MaxPool2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2,2),
    Conv2D(128, 3, activation='relu'),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

model.summary()
#%%
train_datagen = ImageDataGenerator(rescale= 1/255)
train_gen = train_datagen.flow_from_directory(
    './h-s',
    target_size = (150,150),
    class_mode='binary',
    batch_size=32
)
#%%
history = model.fit_generator(train_gen,steps_per_epoch=(80/32), epochs= 50, callbacks=[callback])
#%%