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
nltk.download('punkt')
nltk.download('stopwords')
book_filenames = sorted(glob.glob('*.txt'))

corpus_raw = u""

for book_filename in book_filenames:
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()

print('Corpus is now {0} characters long'.format(len(corpus_raw)))

tokenizer = nltk.data.load()