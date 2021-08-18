#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import zipfile
import shutil
import os
import random
from tensorflow.keras.applications.inception_v3 import InceptionV3
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%%
inception_weights = './Model_saves/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
local_weights_file = inception_weights
pre_trained_model = InceptionV3(
    input_shape=(150,150,3),
    include_top=False,
    weights=None
)
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable =False

pre_trained_model.summary()
#%%
last_layer = pre_trained_model.get_layer('mixed7')
last_out = last_layer.output
#%%
class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['acc'] > 0.99:
            self.model.stop_training = True

#%%
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_out)
x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

model.summary()

#%%
path_hh = './DataSets/horse-or-human.zip'
pat_valhh = './DataSets/validation-horse-or-human.zip'
shutil.rmtree('./DataSets/horse-v-human')
local_zip = path_hh
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./DataSets/horse-v-human/training')
zip_ref.close()

local_zip = pat_valhh
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./DataSets/horse-v-human/validation')
zip_ref.close()
#%%
train_dir = './DataSets/horse-v-human/training'
validation_dir = './DataSets/horse-v-human/validation'

train_horses_dir = train_dir+'/horses/'
train_humans_dir = train_dir+'/humans/'
validation_horses_dir =validation_dir+'/horses'
validation_humans_dir = validation_dir+'/humans'

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))
#%%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'

)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=10,
    target_size=(150, 150),
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=10,
    target_size=(150, 150),
    class_mode='binary'
)

#%%
callback = MyCallBack()
hist = model.fit_generator(train_generator, epochs =10, validation_data=validation_generator, callbacks=[callback])
#%%
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
#%%
