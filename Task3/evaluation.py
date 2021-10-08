from sklearn.model_selection import KFold

from segmentation import Segmentation
from postag import Postag
from utils import *


if __name__ == '__main__':
    n_fold = KFold(5, shuffle=True, random_state=5)

    print('Testing segmentation...')
    segmentation = Segmentation()
    X, Y = load_data()
    Y = Y[:, 0]
    P_segmentation, R_segmentation, F_segmentation = [], [], []
    for train_index, test_index in n_fold.split(X, Y):
        segmentation.fix(X[train_index], Y[train_index])

        res = segmentation.predict(X[test_index])
        p, r, f = prf_segmentation(res, Y[test_index])
        P_segmentation.append(p)
        R_segmentation.append(r)
        F_segmentation.append(f)

    print('Testing posteg...')
    postag = Postag()
    _, Y = load_data()
    X, Y = Y[:, 0], Y[:, 1]
    P_postag, R_postag, F_postag = [], [], []
    for train_index, test_index in n_fold.split(X, Y):
        postag.fix(X[train_index], Y[train_index])

        res = postag.predict(X[test_index])
        p, r, f = prf_postag(res, Y[test_index])
        P_postag.append(p)
        R_postag.append(r)
        F_postag.append(f)
    print('Segmentation result:')
    print(f'P:\t{sum(P_segmentation) / len(P_segmentation) * 100:.3f}%')
    print(f'R:\t{sum(R_segmentation) / len(R_segmentation) * 100:.3f}%')
    print(f'F1:\t{sum(F_segmentation) / len(F_segmentation) * 100:.3f}%')
    print('Postag result:')
    print(f'P:\t{sum(P_postag) / len(P_postag) * 100:.3f}%')
    print(f'R:\t{sum(R_postag) / len(R_postag) * 100:.3f}%')
    print(f'F1:\t{sum(F_postag) / len(F_postag) * 100:.3f}%')
