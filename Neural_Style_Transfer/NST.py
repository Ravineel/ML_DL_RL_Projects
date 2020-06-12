import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt



model  =VGG19(

include_top=False,   # we dont need the top layer, 
                     # only the output of some of the intermediate layers as features
weights="imagenet"  # we want our Model that was trained on imagenet dataset

)
model.trainable =False # we dont want model to update its parametre, we want the model only for output


model.summary()



def img_process(img_path):
    img =load_img(img_path)
    img = img_to_array(img)#so that model could understand img  
    img = preprocess_input(img)# make it suitable for vgg19 input
    img = np.expand_dims(img,axis=0)# expand diminsions of image array 3D to 4D
    return img



def deprocess(arr):
    arr[:,:,0] += 103.939
    arr[:,:,1] += 116.799
    arr[:,:,2] += 123.68

    arr=arr[:,:,::-1]
    arr=np.clip(arr,0,255).astype('uint8')

    return arr

def disp_img(image):
    if len(image.shape)==4:
        img = np.squeeze(image,axis=0)

    img =deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return

disp_img(img_process('style.jpg'))
disp_img(img_process('content.jpg'))








