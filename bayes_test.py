# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets

class bayes_model():
    def __int__(self):
        pass
    def load_data(self):
        data = datasets.load_iris()
        iris_target = data.target
        iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
        train_x, test_x, train_y, test_y = train_test_split(iris_features, iris_target, test_size=0.3, random_state=123)
        return train_x, test_x, train_y, test_y
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf
    def proba_data(self, clf, test_x, test_y):
        y_pred = clf.predict(test_x)
        y_proba = clf.predict_proba(test_x)
        accuracy = metrics.accuracy_score(test_y, y_pred) * 100
        tot1 = pd.DataFrame([test_y, y_pred]).T
        tot2 = pd.DataFrame(y_proba).applymap(lambda x: '%.2f' % x)
        tot = pd.merge(tot1, tot2, left_index=True, right_index=True)
        tot.columns=['y_true', 'y_predict', 'predict_0', 'predict_1', 'predict_2']
        print('The accuracy of Testset is: %d%%' % (accuracy))
        print('The result of predict is: \n', tot.head())
        return accuracy, tot
    def exc_p(self):
        train_x, test_x, train_y, test_y = self.load_data()
        clf = self.train_model(train_x, train_y)
        res = self.proba_data(clf, test_x, test_y)
        return res


if __name__ == '__main__':
    bayes_model().exc_p()