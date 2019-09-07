import gensim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab = model.vocab.keys()

# ws353
dataset = pd.read_csv('./ws353/combined.csv')

# rg65
dataset = pd.read_csv('./rg65/EN-RG-65.txt', sep='\t', header=None)

words = dataset.iloc[:, 0].values

w = []

for each in words:
    if each not in w:
        w.append(each)

words = dataset.iloc[:, 1].values

for each in words:
    if each not in w:
        w.append(each)

count = 0
words_present = []

for each in w:
    if each in vocab:
        count += 1
        words_present.append(each)

word_vectors = []
for each in w:
    word_vectors.append(model[each])


cs = []
for i in range(len(dataset.values)):
    x = model[dataset.values[i][0]]
    y = model[dataset.values[i][1]]
    x = np.reshape(x, (300, 1))
    y = np.reshape(y, (300, 1))
    cs.append(cosine_similarity(x.T, y.T)[0][0])

cs = np.array(cs)
human_cs = dataset.iloc[:, 2].values

# Pearson Correlation
pearson_corr = pearsonr(cs, human_cs)

# Spearman Correlation
spearman_corr = spearmanr(cs, human_cs)