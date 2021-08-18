#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import io
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Flatten

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
#%%
plt.imshow(x_train[0])
logdir = "logs/hand_mnist/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the reshaped image.
with file_writer.as_default():
    images = np.reshape(x_train[0:25], (-1, 28, 28, 1))
    tf.summary.image("25 training data examples", images, max_outputs=25, step=0)


#%%
x_train,x_test = x_train/255.0 , x_test/255.0



#%%
model =Sequential([

    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#%%
hist = model.fit(x_train, y_train, epochs=5)
#%%
model.evaluate(x_test,  y_test, verbose=2)
#%%
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
#%%
print(probability_model(x_test[:5]))
print(np.argmax(probability_model(x_test[:5])))