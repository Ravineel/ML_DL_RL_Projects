#%%
#Libraries

import tensorflow as tf
from tensorflow.keras import models,layers,datasets
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dropout

#%%
# downlaod data
(train_images,train_labels),(test_iamges,test_labels) = datasets.cifar10.load_data()

# normalize image in range 0-1
train_images,test_iamges = train_images/255.0, test_iamges/255.0

#%%
class_names = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']

plt.figure(figsize=(15,15))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# %%
model = models.Sequential([

    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPool2D((2,2)),

    layers.Conv2D(64,3,activation='relu'),
   
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(128,3,activation='relu'),

    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

model.summary()
#%%

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('val_acc')>=0.95):
            print("Accuracy reached 95 percent or more")
            self.model.stop_training=True


callbacks = myCallBack()

hist = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_iamges,test_labels),
    callbacks=[callbacks]
)
# %%
plt.plot(hist.history['acc'],label='accuracy')
plt.plot(hist.history['val_acc'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

