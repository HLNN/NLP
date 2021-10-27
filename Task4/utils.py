import os
import numpy as np
from nltk.translate import AlignedSent
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


def load_data(target='./data/cn.txt', source='./data/en.txt', stem=None):
    print('Loading data...')
    path = f'./data/corpus{"_" + stem if stem else ""}.npy'
    if os.path.exists(path):
        print('Loading data from cache...')
        return np.load(path, allow_pickle=True)
    else:
        print('Loading data from file...')
        corpus = _load_data_from_file(target, source, stem)
        np.save(path, corpus)
        return corpus


def _load_data_from_file(target='./data/cn.txt', source='./data/en.txt', stem=None):
    T, S = [], []
    if stem:
        stemmers = {
            'ps': PorterStemmer(),
            'ls': LancasterStemmer(),
            'ss': SnowballStemmer('en'),
        }
        assert stem in stemmers.keys(), f'Stem should in {stemmers.keys()}'
        stemmer = stemmers[stem]
    else:
        stemmer = None

    def preprocess(line, stemmer=None):
        line = line.lower()
        line = word_tokenize(line)
        if stemmer:
            line = [stemmer.stem(word) for word in line]
        return line

    with open(target, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        T = [preprocess(line) for line in lines]
    with open(source, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        S = [preprocess(line, stemmer=stemmer) for line in lines]

    res = [AlignedSent(t, s) for t, s in zip(T, S)]
    return res


if __name__ == '__main__':
    data = load_data()
    data = load_data(stem='ps')
    data = load_data(stem='ls')
    data = load_data(stem='ss')
    pass
