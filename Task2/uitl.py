import os
import string
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


nltk.download('stopwords')
stop_word = set(stopwords.words('english'))


def download():
    if os.path.exists('./data') and os.path.isdir('./data'):
        return
    url = 'http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes/20_newsgroups.tar.gz'
    os.system(f'wget -O tmp.tar.gz {url}')
    os.system('gunzip tmp.tar.gz && tar -xf tmp.tar && mv 20_newsgroups data')
    os.system('rm tmp.tar.gz tmp.tar')


def load_data(force_disk=False, sw=True, stem='ps'):
    if not os.path.exists('./data'):
        download()

    file_X = './data/' + ('sw' if sw else '') + '_' + str(stem) + '_X.npy'
    file_y = './data/' + ('sw' if sw else '') + '_' + str(stem) + '_y.npy'
    stemmer = {
        'ps': PorterStemmer(),
        'ls': LancasterStemmer(),
        'ss': SnowballStemmer('english'),
        None: None,
    }

    if not force_disk and os.path.exists(file_X) and os.path.exists(file_y):
        print('Loading data from cache!')
        return np.load(file_X, allow_pickle=True), np.load(file_y, allow_pickle=True)
    else:
        print('Loading data from origin file...')
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
                    data = preprocess(data, sw, stemmer.get(stem))
                    x.append(data)
            X.append(np.array(x))

        X, y = np.concatenate(X).reshape(-1, 1), np.concatenate(y).reshape(-1, 1)
        np.save(file_X, X)
        np.save(file_y, y)
        return X, y


def preprocess(s, sw, stem):
    # remove_punctuation
    s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    s = re.split(r'\W+', s)
    # stop words and stemming
    s = [stem.stem(word) if stem else word for word in s if sw and (word not in stop_word)]
    return s
