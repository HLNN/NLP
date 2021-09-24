from collections import defaultdict
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.k = 0
        self.prior = None
        self.words = None

    def clean(self):
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
        for i, x in enumerate(X):
            print(f'\r[{i} / {len(X)}]', end='')
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

            category_profile = np.log(category_profile)
            P = np.dot(input_vector, category_profile)
            pred.append(P)
        print()
        return np.concatenate(pred).argmax(axis=1).reshape(-1, 1)
