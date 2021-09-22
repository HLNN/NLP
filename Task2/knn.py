from collections import defaultdict, Counter
import heapq
import numpy as np


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
    def __init__(self, n=20):
        self.heap = Heap(n)
        self.X = None
        self.y = None

    def count(self, x):
        wc = defaultdict(int)
        for word in x:
            wc[word] += 1
        return wc

    def dist(self, x, vector):
        res = .0
        for word in x.keys():
            res += float(x[word] - vector[word]) ** 2
        return res ** 0.5

    def fix(self, X, y):
        self.X = np.array([self.count(x[0]) for x in X]).reshape(-1, 1)
        self.y = y

    def predict(self, X):
        if self.X is None:
            raise Exception('you have to fit first before predict')

        pred = []
        for i, x in enumerate(X):
            print(f'\r[{i} / {len(X)}]', end='')
            words = defaultdict(int)
            for word in x[0]:
                words[word] += 1

            self.heap.clean()
            for vector, group in zip(self.X, self.y):
                self.heap.insert(self.dist(words, vector[0]), group[0])

            min_dist = self.heap.get()
            pred.append(Counter(min_dist).most_common(1)[0][0])

        print()
        return np.array(pred).reshape(-1, 1)
