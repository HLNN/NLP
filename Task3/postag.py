import numpy as np

from hmm import HMM
from utils import *


class Postag(HMM):
    def __init__(self):
        super().__init__()

    def predict(self, X):
        if isinstance(X, str):
            X = X.strip().split()
        if isinstance(X, list):
            X = np.array([X, []])[:-1]
        return super().predict(X)


if __name__ == '__main__':
    postag = Postag()
    _, Y = load_data()
    X, Y = Y[:, 0], Y[:, 1]
    postag.fix(X, Y)

    n = 100
    print('Predict:')
    res = postag.predict(X[:n])
    print(res[0])
    print('Gt:')
    print(Y[0])
    P, R, F = prf_postag(res, Y[:n])
    print(f'P:\t{P * 100:.3f}%')
    print(f'R:\t{R * 100:.3f}%')
    print(f'F1:\t{F * 100:.3f}%')
