
#%%
import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


#%%
model  =VGG19(

include_top=False,   # we dont need the top layer, 
                     # only the output of some of the intermediate layers as features
weights="imagenet"  # we want our Model that was trained on imagenet dataset

)
model.trainable =False # we dont want model to update its parametre, we want the model only for output


model.summary()


#%%
def img_process(img_path):
    img =load_img(img_path)
    img = img_to_array(img)#so that model could understand img  
    img = preprocess_input(img)# make it suitable for vgg19 input
    img = np.expand_dims(img,axis=0)# expand diminsions of image array 3D to 4D as models require 4D tensor
    return img


#%%
def deprocess(arr):
    arr[:,:,0] += 103.939
    arr[:,:,1] += 116.779
    arr[:,:,2] += 123.68

    arr=arr[:,:,::-1]
    arr=np.clip(arr,0,255).astype('uint8')

    return arr
#%%
def disp_img(image):
    if len(image.shape)==4:
        img = np.squeeze(image,axis=0)

    img =deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return
#%%
disp_img(img_process('style.jpg'))
#%%
disp_img(img_process('content.jpg'))
#%%

content_layer = 'block5_conv2'

style_layers=['block1_conv1','block2_conv1','block3_conv1']

content_model = Model(inputs = model.input,
                        outputs = model.get_layer(content_layer).output

                        )

style_models = [ Model(inputs= model.input, 
    outputs= model.get_layer(layer).output)   for layer in style_layers]

# %%
def content_cost(content,generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C-a_G))
    return cost

# %%
def gram_matrix(A):
    n_C = int(A.shape[-1])
    a = tf.reshape(A, [-1,n_C])
    n = tf.shape(a)[0]
    G = tf.matmul(a,a,transpose_a=True)
    return G/ tf.cast(n, tf.float32)

# %%
lam = 1./len(style_models)

def style_cost(style,generated):
    J_style = 0

    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS-GG))
        J_style += current_cost*lam
    
    return J_style

# %%
import time

def training_loop(content_path,style_path, iterations =20,
                alpha =10., beta =20.):
    
    content = load_img(content_path)
    style = load_img(style_path)

    generated  = tf.Variable(content, dtype=tf.float32)

    opt = tf.train.AdamOptimizer(learning_rate =7.)
    
    best_cost = 1e12 +0.1
    best_image = None

    start_time = time.time()
    generated_images = []
    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_cost(content,generated)
            J_style = style_cost(style,generated)
            J_total = alpha*J_content + beta* J_style

        grads = tape.gradient(J_total,generated)
        opt.apply_gradients([(grads,generated)])

        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.np()
        
        print('Cost at {}:{}. Time elapsed: {}'.format(i,J_total,time.time()-start_time))
        
        generated_images.append(generated.np())

    return best_image

# %%
best_image = training_loop('content.jpg','style.jpg')

<<<<<<< HEAD
# %%
disp_img(best_image)


# %%
=======
>>>>>>> parent of c6694cba27... ML-Udacity Class
