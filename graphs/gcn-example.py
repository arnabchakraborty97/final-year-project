import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from networkx import karate_club_graph, to_numpy_matrix

zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))

A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

# Relu function
def relu(a):
    return np.maximum(a, 0, a)

def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

output = H_2

feature_representations = {
    node: np.array(output)[node] 
    for node in zkc.nodes()}

o = list(np.array(output))