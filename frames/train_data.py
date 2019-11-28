
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.callbacks import TensorBoard
from seaborn import heatmap

# Loading back the data

import pickle

with open("X.pkl", "rb") as handle:
    X = pickle.load(handle)
with open("Y.pkl", "rb") as handle:
    Y = pickle.load(handle)
with open("classes.pkl", "rb") as handle:
    classes = pickle.load(handle)
    
X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Creating the model
model = tf.keras.Sequential()

dense_num = 0
conv_num = 3
layers = 64

# First Convolutional Layer
model.add(Conv2D(layers, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Remaining Convolutional Layers
for _ in range(conv_num-1):
    model.add(Conv2D(layers, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

# 3D feature maps to 1D feature vector
model.add(Flatten())

# Dense Layers
for _ in range(dense_num):
    model.add(Dense(layers))
    model.add(Activation('relu'))

# Final Dense Layer to shrink to 3 ouput
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, 
              optimizer=tf.keras.optimizers.Adam(), 
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/CNN-Trained")

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[tensorboard])

model.evaluate(X_test, y_test)

predictions = model.predict(X_test)

y_pred = []

for p in predictions:
    y_pred.append(np.argmax(p))

y_pred = np.array(y_pred)

heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

model.save("model.h5")

model.predict([X_test[0].reshape(-1, 40, 50, 1)])


import cv2
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 40), interpolation=cv2.INTER_AREA)
    
    resized = resized / 255.0
    prediction = classes[np.argmax(model.predict([resized.reshape(-1, 40, 50, 1)]))]
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 400)
    fontScale              = 3
    fontColor              = (255,255,255)
    lineType               = 10

    cv2.putText(img, prediction, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()