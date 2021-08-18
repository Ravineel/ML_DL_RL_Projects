import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model,layers,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%%
def get_data(filename):
    with open(filename) as training_file:
        file = csv.reader(training_file, delimiter = ',')
        c = 0
        image = []
        label = []

        for row in file:
            if c==0:
                c+=1
                continue
            label.append(row[0])
            image_data = row[1:785]
            x = np.array_split(image_data,28)
            image.append(x)
        labels = np.array(label).astype('float')
        images = np.array(image).astype('float')

        return images,labels

path_sign = './DataSets/sign_mnist_train.csv'
path_val_sign = './DataSets/sign_mnist_test.csv'

training_image, training_labels = get_data(path_sign)
testing_image, testing_labels = get_data(path_val_sign)
#%%
print(training_image.shape)
print(training_labels.shape)
print(testing_image.shape)
print(testing_labels.shape)
#%%
training_images = np.expand_dims(training_image, axis=3)  # Your Code Here
testing_images = np.expand_dims(testing_image, axis=3)  # Your Code Here


train_datagen = ImageDataGenerator(

    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

print(training_images.shape)
print(testing_images.shape)
#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics = ['acc'])

model.summary()

history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=10),
                              epochs=10,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=10)
                              )
model.save('./Model_saves/Sign/sign.h5')

#%%
acc = history.history['acc']# Your Code Here
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
#%%