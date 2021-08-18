#%%

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import zipfile
import os
import random
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%
path = './Datasets/cats-and-dogs.zip'
zip = zipfile.ZipFile(path,'r')
zip.extractall("./DataSets/cats-dog")
zip.close()
#%%
print(len(os.listdir('./DataSets/cats-dog/PetImages/Cat')))
print(len(os.listdir('./DataSets/cats-dog/PetImages/Dog')))
#%%
try:
    os.mkdir('./DataSets/cats-dog/CatvDog/')
    os.mkdir('./DataSets/cats-dog/CatvDog/training/')
    os.mkdir('./DataSets/cats-dog/CatvDog/testing/')
    os.mkdir('./DataSets/cats-dog/CatvDog/training/cats/')
    os.mkdir('./DataSets/cats-dog/CatvDog/training/dogs/')
    os.mkdir('./DataSets/cats-dog/CatvDog/testing/cats/')
    os.mkdir('./DataSets/cats-dog/CatvDog/testing/dogs/')


except OSError:
    pass
#%%
def split_data(SOURCE, TRAINING, TESTING, SPILT_SIZE):
    filesname = os.listdir(SOURCE)
    files=[]
    for f in filesname:
        if os.path.getsize(SOURCE+f)!=0:
            files.append(f)

    n= len(files)
    shuffle_list = random.sample(files,n)
    up_limit = int(n*SPILT_SIZE)
    training_list = shuffle_list[:up_limit]
    testing_list = shuffle_list[up_limit:]

    for f in training_list:
        copyfile(SOURCE+f,TRAINING+f)
    for f in testing_list:
        copyfile(SOURCE+f,TESTING+f)


cats_source_dir ='./DataSets/cats-dog/PetImages/Cat/'
dog_source_dir = './DataSets/cats-dog/PetImages/Dog/'
Training_cat = './DataSets/cats-dog/CatvDog/training/cats/'
Testing_cat = './DataSets/cats-dog/CatvDog/testing/cats/'
Training_dog = './DataSets/cats-dog/CatvDog/training/dogs/'
Testing_dog = './DataSets/cats-dog/CatvDog/testing/dogs/'
split_size = 0.9
split_data(cats_source_dir, Training_cat,Testing_cat,split_size)
split_data(dog_source_dir, Training_dog, Testing_dog, split_size)
#%%
print(len(os.listdir('./DataSets/cats-dog/CatvDog/training/cats/')))
print(len(os.listdir('./DataSets/cats-dog/CatvDog/training/dogs/')))
print(len(os.listdir('./DataSets/cats-dog/CatvDog/testing/cats/')))
print(len(os.listdir('./DataSets/cats-dog/CatvDog/testing/dogs/')))
#%%
model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(150, 150, 3)),
    MaxPool2D(2,2),
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
#%%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)

train_dir = './DataSets/cats-dog/CatvDog/training'
val_dir = './DataSets/cats-dog/CatvDog/testing'

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size= 32,
    class_mode ='binary'

)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
#%%
class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] > 0.95:
            self.model.stop_training= True

stopcall = MyCallBack()
check = tf.keras.callbacks.ModelCheckpoint('./models/catvdog/', save_best_only=True )
earlstop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

#%%
#model = tf.keras.models.load_model('./Model_saves/Cat-Dog/mymodel.h5')
hist = model.fit_generator(train_gen, epochs = 1, validation_data=val_gen, callbacks=[stopcall,check,earlstop])
model.save('./Model_saves/Cat-Dog/mymodel.h5')
#%%
import matplotlib.image  as mpimg

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=hist.history['acc']
val_acc=hist.history['val_acc']
loss=hist.history['loss']
val_loss=hist.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.show()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')
plt.show()
