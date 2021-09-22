import os
import string
import re
from collections import defaultdict
import numpy as np


def download():
    if os.path.exists('./data') and os.path.isdir('./data'):
        return
    url = 'http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes/20_newsgroups.tar.gz'
    os.system(f'wget -O tmp.tar.gz {url}')
    os.system('gunzip tmp.tar.gz && tar -xf tmp.tar && mv 20_newsgroups data')
    os.system('rm tmp.tar.gz tmp.tar')


def load_data():
    if not os.path.exists('./data'):
        download()

    X, y = [], []

    for parent, dirnames, filenames in os.walk('./data'):
        if parent == './data':
            groups_name = {i: name for i, name in enumerate(dirnames)}
            groups_id = {name: i for i, name in enumerate(dirnames)}
            continue

        group = parent.split('/')[-1]
        y.append(np.array([groups_id[group]] * len(filenames)))

        x = []
        for filename in filenames:
            with open(f'{parent}/{filename}', 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
                data = preprocess(data)
                x.append(data)
        X.append(np.array(x))

    return np.concatenate(X).reshape(-1, 1), np.concatenate(y).reshape(-1, 1)


def preprocess(s):
    # remove_punctuation
    s.translate(str.maketrans('', '', string.punctuation))
    # stop words
    # TODO
    # words stemming
    # TODO
    s = s.lower()
    s = re.split(r'\W+', s)
    return s


def count_words(s):
    wc = defaultdict(int)
    for w in s:
        w[w] += 1
    return wc


class NaiveBayes:
    def __init__(self):
        self.k = 0
        self.prior = None
        self.words = None

    def fix(self, X, y):
        # Prior
        self.k = y.max() + 1
        self.prior = np.array([sum(np.squeeze(y) == i) for i in range(self.k)])
        # self.prior += np.ones_like(self.prior)
        self.prior = (self.prior / sum(self.prior)).reshape(1, -1)

        self.words = [defaultdict(int) for _ in range(self.k)]
        for x, group in zip(X, y):
            for word in x[0]:
                self.words[group[0]][word] += 1

    def predict(self, X):
        if self.prior is None:
            raise Exception('you have to fit first before predict')

        pred = []
        N = len(X)
        for x in X:
            words = defaultdict(int)
            for word in x[0]:
                words[word] += 1

            M = len(words)

            input_vector = np.ones((1, M+1))
            category_profile = []

            for i, (word, count) in enumerate(words.items()):
                input_vector[0, i] = count

                category_likehood = np.array([self.words[cate][word] for cate in range(self.k)]).reshape(1, -1)
                category_likehood += 1
                category_likehood = category_likehood / np.sum(category_likehood, axis=1)
                category_profile.append(category_likehood)
            category_profile = np.concatenate(category_profile + [self.prior])

            P = np.dot(input_vector, category_profile)
            pred.append(P)
        return np.concatenate(pred).argmax(axis=1).reshape(-1, 1)


if __name__ == '__main__':
    X, y = load_data()
    nb = NaiveBayes()
    nb.fix(X, y)
    res = nb.predict(X)
    rate = sum(res == y) / len(y)
    print(rate)
    pass
