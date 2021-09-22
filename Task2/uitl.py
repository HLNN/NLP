import os
import string
import re
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
