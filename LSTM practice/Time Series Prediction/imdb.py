# imdb reviews

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(7)

# load dataset
top_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = top_words)

# truncate and pad input sequences
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen = max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen = max_review_length)

# Creating the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length = max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3, batch_size = 64, validation_data = (x_test, y_test))