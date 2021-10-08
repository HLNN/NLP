import numpy as np

from hmm import HMM
from utils import *


class Segmentation(HMM):
    def __init__(self):
        super().__init__()

    def __tag(self, word):
        if len(word) == 1:
            state = 'S'
        else:
            state = f'B{"M" * (len(word) - 2)}E'
        return state

    def __text(self, res, X):
        cut_res = []
        for state, text in zip(res, X):
            text_cut, begin = [], 0
            for i, ch in enumerate(text):
                if state[i] == 'B':
                    begin = i
                elif state[i] == 'E':
                    text_cut.append(text[begin:i + 1])
                elif state[i] == 'S':
                    text_cut.append(text[i])
            if state[-1] == 'M':
                text_cut.append(text[begin:])
            cut_res.append(text_cut)
        return np.array(cut_res + [[]])[:-1]

    def fix(self, X, Y):
        Y = np.array([''.join(self.__tag(word) for word in line) for line in Y])
        super().fix(X, Y)

    def predict(self, X):
        if isinstance(X, str):
            X = np.array([X])

        res = super().predict(X)
        return self.__text(res, X)


if __name__ == '__main__':
    s = Segmentation()
    X, Y = load_data()
    Y = Y[:, 0]
    s.fix(X, Y)

    n = 1000
    print('Predict:')
    res = s.predict(X[:n])
    print(res[0])
    print('Gt:')
    print(Y[0])
