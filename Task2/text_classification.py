from sklearn.model_selection import KFold

from knn import KNN
from naive_bayes import NaiveBayes
from uitl import *


class NFold:
    def __init__(self, classifier, n=200):
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


if __name__ == '__main__':
    classifier_name = 'knn'

    if classifier_name == 'naive_bayes':
        classifier = NaiveBayes()
    elif classifier_name == 'knn':
        classifier = KNN()
    else:
        assert False, 'Classifier: naive_bayes or knn'

    nfold = NFold(classifier)
    nfold.run()
    pass
