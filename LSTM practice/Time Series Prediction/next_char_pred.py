# Predict next character

import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation, Dropout, TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import heapq

np.random.seed(42)

# Load dataset
text = open('../../word2vec practice/Game of thrones/Game of Thrones.txt').read().lower()

# store unique characters
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Preprocess data
SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

# generate features and labels    
x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
# Build the model
model = Sequential()
model.add(LSTM(128, input_shape = (SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

model.fit(x, y, validation_split = 0.05, batch_size = 128, epochs = 20, shuffle = True)
    
