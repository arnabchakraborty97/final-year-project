# Word2Vec

import codecs
import glob
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

book_filenames = sorted(glob.glob("*.txt"))

corpus_raw = u""
for book_filename in book_filenames:
    with codecs.open(book_filename, "r") as book_file:
        corpus_raw += book_file.read()
        
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if (len(raw_sentence) > 0):
        sentences.append(sentence_to_wordlist(raw_sentence))
        
# Train Word2Vec

# 3 main tasks
# Distance, Similarity, Ranking

# Dimensionality of the resulting word vectors
num_features = 300

# Minimum word count threshold
min_word_count = 3

# Number of threads to run in parallel
num_workers = multiprocessing.cpu_count()

# Context window length
context_size = 7

# Downsample setting for frequent words
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible
seed = 1

harry2vec = w2v.Word2Vec(
            sg = 1,
            seed = seed,
            workers = num_workers,
            size = num_features,
            min_count = min_word_count,
            window = context_size,
            sample = downsampling
        )

harry2vec.build_vocab(sentences)

harry2vec.train(sentences, total_examples = harry2vec.corpus_count, epochs = harry2vec.epochs)

if not os.path.exists('trained'):
    os.makedirs('trained')
harry2vec.save(os.path.join('trained', 'harry2vec.w2v'))


# Explore the trained model
harry2vec = w2v.Word2Vec.load(os.path.join('trained', 'harry2vec.w2v'))

# Compress the word vectors into 2D space and plot them
tsne = sklearn.manifold.TSNE(n_components = 2, random_state = 0)
all_word_vectors_matrix = harry2vec.syn1neg
all_word_vectors_matrix_2D = tsne.fit_transform(all_word_vectors_matrix)

# Plotting the big picture
points = pd.DataFrame(
            [
                    (word,  coords[0], coords[1])
                    for word, coords in [
                            (word, all_word_vectors_matrix_2D[i])
                            for i, word in enumerate(harry2vec.wv.vocab)
                        ]
            ],
            columns = ['word', 'x', 'y']
        )
    
sns.set_context('poster')

points.plot.scatter('x', 'y', s = 10, figsize = (10, 6))
    
def plot_region(x_bounds, y_bounds):
    slice = points[
                (x_bounds[0] <= points.x) &
                (points.x <= x_bounds[1]) &
                (y_bounds[0] <= points.y) &
                (points.y <= y_bounds[1])
            ]
    
    ax = slice.plot.scatter('x', 'y', s = 35, figsize = (6, 4))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize = 1)
        
plot_region(x_bounds = (30, 42), y_bounds = (10, 20))

harry2vec.wv.most_similar("Magic")