import numpy as np
import matplotlib.pyplot as ply
import pandas as pd
import keras

#Relu function
def relu(a):


# Adjacency Matrix
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)

# Feature Matrix
X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float)

# Adding self loops
I = np.matrix(np.eye(A.shape[0]))

A_hat = A + I

# Normalizing
D = np.array(np.sum(A, axis=0))[0]
D = np.matrix(np.diag(D))

D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

# Putting it together
W = np.matrix([
             [1, -1],
             [-1, 1]
         ])

#A complete hidden layer with adjacency matrix, input features, weights and activation function!
np.maximum(D_hat**-1 * A_hat * X * W, 0, D_hat**-1 * A_hat * X * W)



