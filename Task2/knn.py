from collections import defaultdict, Counter
import heapq
import numpy as np
import math


class Heap:
    def __init__(self, n):
        self.n = n
        self.count = 0
        self.q = []

    def insert(self, dist, item):
        self.count += 1
        if len(self.q) < self.n:
            heapq.heappush(self.q, (-dist, self.count, item))
        else:
            heapq.heapreplace(self.q, (-dist, self.count, item))

    def get(self):
        return [item[-1] for item in self.q[::-1]]

    def clean(self):
        self.q, self.count = [], 0


class KNN:
    def __init__(self, k=20):
        self.heap = Heap(k)
        self.X = None
        self.y = None

    def update_k(self, k):
        self.heap = Heap(k)

    def count(self, x):
        wc = defaultdict(int)
        for word in x:
            wc[word] += 1
        return wc

    def dist(self, x, vector, dist_type='euclidean'):
        res = .0
        if dist_type == 'euclidean':
            # dist = numpy.linalg.norm(a-b)
            for word in x.keys():
                res += float(x[word] - vector[word]) ** 2
            return res ** 0.5
        elif dist_type == 'cos':
            x1, x2 = np.zeros((len(x),)), np.zeros((len(x),))
            for i, word in enumerate(x.keys()):
                x1[i], x2[i] = x[word], vector[word]
            return np.sum(x1 * x2) / (np.sqrt(np.sum(np.square(x1))) + np.sqrt(np.sum(np.square(x2))))
        else:
            assert False, f'Unknown dist type: {dist_type}'

    def choose_k(self, X):
        if self.X is None:
            raise Exception('you have to fit first before choose_k')

        pred = []
        for i, x in enumerate(X):
            print(f'\r[{i} / {len(X)}]', end='')
            words = defaultdict(float)
            for word in x[0]:
                words[word] += 1

            # Try to speed up predict, but slower
            # Now use for choose better k
            input_vector = np.zeros((len(words),))
            text_vector = np.zeros((len(self.X), len(words)))

            for word_index, (word, count) in enumerate(words.items()):
                input_vector[word_index] = count
                for text_index, text in enumerate(self.X):
                    text_vector[text_index, word_index] = text[0][word]

            # Ignore sqrt
            dist = np.sum(np.square(text_vector - input_vector), axis=1)
            dist_pair = [(d, group[0]) for d, group in zip(dist, self.y)]
            dist_pair.sort(key=lambda d: d[0])

            pred.append(np.array([pair[1] for pair in dist_pair]))

        print()
        return np.array(pred)

    def fix(self, X, y, tf_idf=False):
        self.X = np.array([self.count(x[0]) for x in X]).reshape(-1, 1)
        self.y = y

        if tf_idf:
            N = len(self.X)
            n = defaultdict(float)
            for x in self.X:
                for word in x[0].keys():
                    n[word] += 1

            for key in n.keys():
                n[key] = math.log(N / n[key])

            for i, x in enumerate(self.X):
                for word in x[0].keys():
                    x[0][word] *= n[word]

    def predict(self, X):
        if self.X is None:
            raise Exception('you have to fit first before predict')

        pred = []
        for i, x in enumerate(X):
            print(f'\r[{i} / {len(X)}]', end='')
            words = defaultdict(float)
            for word in x[0]:
                words[word] += 1

            self.heap.clean()
            for vector, group in zip(self.X, self.y):
                self.heap.insert(self.dist(words, vector[0]), group[0])

            min_dist = self.heap.get()
            pred.append(Counter(min_dist).most_common(1)[0][0])

        print()
        return np.array(pred).reshape(-1, 1)
