from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from knn import KNN
from naive_bayes import NaiveBayes
from uitl import *


class NFold:
    def __init__(self, classifier, n=5):
        self.n_fold = KFold(n, shuffle=True, random_state=5)
        self.classifier = classifier
        self.X, self.y = load_data()

    def run(self):
        for i, (train_index, test_index) in enumerate(self.n_fold.split(self.X, self.y)):
            X_train, y_train = self.X[train_index], self.y[train_index]
            self.classifier.fix(X_train, y_train)

            X_test, y_test = self.X[test_index], self.y[test_index]
            res = self.classifier.predict(X_test)
            print(i, sum(res == y_test) / len(y_test))

            cm = confusion_matrix(y_test, res)
            print(cm)


class HoldOut:
    def __init__(self, classifier, train_val_test=(.96, .02, .02), choose_k=False):
        if sum(train_val_test) != 1.:
            assert False
        self.train_test = train_val_test[0]
        self.val_test = train_val_test[1] / sum(train_val_test[1:])
        self.classifier = classifier
        self.choose_k = choose_k
        self.X, self.y = load_data()

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            train_size=self.train_test,
                                                            random_state=5
                                                            )
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                        train_size=self.val_test,
                                                        random_state=5
                                                        )
        self.classifier.fix(X_train, y_train)

        if self.choose_k:
            res = self.classifier.choose_k(X_val)

            counters = [Counter() for _ in range(res.shape[0])]
            p = []
            for k in range(res.shape[1]):
                for i in range(res.shape[0]):
                    counters[i][res[i, k]] += 1

                pred = np.array([c.most_common(1)[0][0] for c in counters]).reshape(-1, 1)
                P = sum(pred == y_val) / len(y_val)
                p.append(P)
            p = np.array(p)
            best_k = np.argmax(p, axis=0)
            print(f'Choose best k: {best_k}')
            plt.plot(p)
            plt.xlabel('k')
            plt.ylabel('precision')
            plt.savefig('Precision.png')

            self.classifier.update_k(best_k[0])

        res = self.classifier.predict(X_test)
        P = sum(res == y_test) / len(y_test)
        print(f'Precision: {P}')
        cm = confusion_matrix(y_test, res)
        print(cm)


if __name__ == '__main__':
    classifier_name = 'knn'
    val_method = 'hold_out'
    k_choose = True

    if classifier_name == 'naive_bayes':
        classifier = NaiveBayes()
    elif classifier_name == 'knn':
        classifier = KNN()
    else:
        assert False, 'Classifier: naive_bayes or knn'

    if classifier_name != 'knn:': k_choose = False
    if val_method == 'nfold':
        nfold = NFold(classifier)
        nfold.run()
    elif val_method == 'hold_out':
        hold_out = HoldOut(classifier, choose_k=True)
        hold_out.run()
    else:
        assert False, 'Val: nfold or hold_out'

    pass
