# Most of the code is borrowed from the Udemy course Advanced NLP and RNNs
# https://deeplearningcourses.com/c/deep-learning-advanced-nlp

from __future__ import print_function, division
from builtins import range

import os
from os import walk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

# some configuration
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3
BATCH_SIZE = 8
EPOCHS = 100

# Download the Fasttext Arabic word vectors from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz
# Unzip and place them in the vectors folders

# load in pre-trained arabic word vectors from fasttext
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('../vectors/cc.ar.300.vec')) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# this is is not fully annotated ABSA data. it comes with aspect categorization only
# sentiment annotation will be added to the aspects in the future
# this is a subset of the corpus prepared for the "Sentiment Analysis and Subjectivity Detection in Arabic Documents" project
# if you use this data in your research please cite the following paper
# Muazzam Ahmed Siddiqui, Mohamed Yehia Dahab, and Omar Abdullah Batarfi. 2015. Building A Sentiment Analysis Corpus With Multifaceted Hierarchical Annotation. International Journal of Computational Linguistics (IJCL) (2015).

# read the data
rawdata = pd.read_csv('..data/absa7_data.csv')

reviews = rawdata['review'].values
possible_labels = ["HOTEL#CLEANLINESS", "HOTEL#COMFORT", "HOTEL#GENERAL", "HOTEL#PRICE", "LOCATION#GENERAL", "SERVICE#GENERAL", "STAFF#GENERAL"]
labels = rawdata[possible_labels].values

# split the data into training and testing
train_reviews, test_reviews, train_targets,  test_targets = train_test_split(reviews, labels, test_size=0.1)

# find the maximum and average review (sequence) length
rev_len = np.zeros(len(train_reviews))
i=0
for r in train_reviews:
    rev_len[i] = len(r.split(' '))
    i = i+1
print(rev_len.max())
# 51
print(rev_len.mean())
# 8.499

MAX_SEQUENCE_LENGTH = 51

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_reviews)
sequences = tokenizer.texts_to_sequences(train_reviews)

# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))


# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector

# word2vec is no longer needed. delete it to save memory
#del(word2vec)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)

print('Building model...')

# create an LSTM network with a single LSTM layer
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = LSTM(50, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy'],
)


print('Training model...')
r = model.fit(
  data,
  train_targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)

# accuracies
# this is work in progress. the learning curves point to overfitting
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# prepare testing data
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(test_reviews)
sequences = tokenizer.texts_to_sequences(test_reviews)

# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))


# pad sequences so that we get a N x T matrix
test_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', test_data.shape)


p = model.predict(test_data)
aucs = []
for j in range(7):
    auc = roc_auc_score(test_targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))
