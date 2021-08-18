#%%
import json
import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt
#%%
embedding_dim =100
max_length =16
trunc_type = 'post'
padd_type = 'post'
oov_token = '<oov>'
training_size =160000
test_partition = 0.1
 #%%
corpus =[]
path_sent = './DataSets/training_cleaned.csv'
num_sentences = 0
with open(path_sent) as csvfile:
    file = csv.reader(csvfile, delimiter =',')
    for row in file:

        list_item = []

        list_item.append(row[5])
        this_label = row[0]
        if this_label=='0':
            list_item.append(0)
        else:
            list_item.append(1)

        num_sentences +=1
        corpus.append(list_item)
#%%



