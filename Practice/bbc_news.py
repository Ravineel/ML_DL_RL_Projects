# %%
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import getcwd

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%
path_bbc = './Datasets/bbc-text.csv'
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]
# %%
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<oov>'
training_portion = .8
# %%
sentences = []
labels = []
with open(path_bbc, 'r') as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    c = 0
    for f in file:
        if c == 0:
            c += 1
            continue
        labels.append(f[0])
        sentence = f[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            # sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
print(len(labels))
print(len(sentences))
print(sentences[0])

print(len(sentences))
print(sentences[0])
# %%
train_size = int(len(sentences) * training_portion)  # YOUR CODE HERE

train_sentences = sentences[:train_size]  # YOUR CODE HERE
train_labels = labels[:train_size]  # YOUR CODE HERE

validation_sentences = sentences[train_size:]  # YOUR CODE HERE
validation_labels = labels[train_size:]  # YOUR CODE HERE

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))
# %%
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))
# %%
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length)

print(len(validation_sequences))
print(validation_padded.shape)
# %%
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

# %%

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# %%
history = model.fit(train_padded, training_label_seq, epochs=30,
                    validation_data=(validation_padded, validation_label_seq))
# %%
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "acc")
plot_graphs(history, "loss")

# %%
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# %%
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
# %%
