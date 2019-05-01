import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from heapq import heappush, heappop
import sys


def split_test_train(X, y, percent=0.8):
    mask = np.random.rand(len(X)) < percent
    X_train = X[mask].dropna()
    X_test = X[~mask].dropna()
    y_train = y[mask].dropna()
    y_test = y[~mask].dropna()
    print(X_train.shape, X_test.shape)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def evaluate_result(y_test, y_pred):
    #     pprint(confusion_matrix(y_test, y_pred))
    #     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #     print("tn = {0}, fp = {1}, fn = {2}, tp = {3}".format(tn, fp, fn, tp))
    f1 = f1_score(y_test, y_pred, average="macro")
    #     print("f1_score\t : ", f1)
    ps = precision_score(y_test, y_pred, average="macro")
    #     print("precision_score\t : ", ps)
    rs = recall_score(y_test, y_pred, average="macro")
    #     print("recall_score\t : ", rs)
    acs = accuracy_score(y_test, y_pred)
    #     print("accuracy_score\t : ", acs)
    return np.array([acs, ps, rs, f1])


def convert_categorical_to_numerical(data):
    res = pd.factorize(data)
    x, y = res
    return x, y

def euclid(test_row, train_row):
    dis = np.sqrt(np.sum([(x - y)**2 for x, y in zip(test_row, train_row)]))
    return dis


def manhattan(test_row, train_row):
    dis = np.sum([abs(x - y) for x, y in zip(train_row, test_row)])
    return dis


def chebyshew(test_row, train_row):
    dis = np.max([abs(x - y) for x, y in zip(train_row, test_row)])
    return dis


class KNN:
    def fit(self, train_X, train_y, k=3, distance_function=euclid):
        self.train_X = train_X
        self.min_X, self.max_X = train_X.min(), train_X.max()
        self.train_X = self.normalize(self.train_X)
        self.train_y = train_y
        self.k = k
        self.distance_function = distance_function

    def normalize(self, X):
        X_normalized = (X - self.min_X) / (self.max_X-self.min_X)
        return X_normalized
        
    def predict_row(self, test_row):
        heap = []
        for (index, train_row) in self.train_X.iterrows():
            dis = -self.distance_function(train_row, test_row)
            x = (dis, self.train_y.iloc[index, 0])
            heappush(heap, x)
            if len(heap) > self.k:
                heappop(heap)
        elem, count = np.unique([i for j, i in heap], return_counts=True)
        return elem[np.argmax(count)]

    def predict(self, test_X):
        test_y = pd.DataFrame()
        test_y['predicted'] = [
            self.predict_row(row) for i, row in test_X.iterrows()
        ]
        return test_y

    def evaluate_result(self, y_test, y_pred):
        #         pprint(confusion_matrix(y_test, y_pred))
        f1 = f1_score(y_test, y_pred, average="macro")
        ps = precision_score(y_test, y_pred, average="macro")
        rs = recall_score(y_test, y_pred, average="macro")
        acs = accuracy_score(y_test, y_pred)
        #         print("accuracy_score\t : ", acs)
        return np.array([acs, ps, rs, f1])

    def set_k(self, k):
        self.k = k

if __name__ == "__main__":
    
    robot1 = pd.read_csv('RobotDataset/Robot1',header=None, delim_whitespace=True)
    robot2 = pd.read_csv('RobotDataset/Robot2',header=None, delim_whitespace=True)
    robot1.columns = robot2.columns = ['class', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'id']
    list(robot1.columns.values)
    robot1 = robot1.drop('Unnamed: 0', 1)
    robot2 = robot2.drop('Unnamed: 0', 1)
    robot1 = robot1.drop('id', 1)
    robot2 = robot2.drop('id', 1)
    iris_data = pd.read_csv("Iris/Iris.csv")

    array, convert_back_Iris_to_categorical = convert_categorical_to_numerical(
        iris_data['class'])
    iris_data['class'] = array

    Iris_X = iris_data.drop('class', 1)
    Iris_y = iris_data[['class']]
    Iris_train_X, Iris_test_X, Iris_train_y, Iris_test_y = split_test_train(
        Iris_X, Iris_y, 0.2)


    robot1_X = robot1.drop('class', 1)
    robot1_y = robot1[['class']]
    robot1_train_X, robot1_test_X, robot1_train_y, robot1_test_y = split_test_train(
        robot1_X, robot1_y, 0.2)


    robot2_X = robot2.drop('class', 1)
    robot2_y = robot2[['class']]
    robot2_train_X, robot2_test_X, robot2_train_y, robot2_test_y = split_test_train(
        robot2_X, robot2_y, 0.2)



    classifier_robot1 = KNN()
    classifier_robot1.fit(robot1_train_X, robot1_train_y, 2)
    y_predict_robot1 = classifier_robot1.predict(robot1_test_X)
    print(robot1_test_y.shape, y_predict_robot1.shape)
    classifier_robot1.evaluate_result(robot1_test_y, y_predict_robot1)


    classifier_robot2 = KNN()
    classifier_robot2.fit(robot2_train_X, robot2_train_y, 2)
    y_predict_robot2 = classifier_robot2.predict(robot2_test_X)
    print(robot2_test_y.shape, y_predict_robot2.shape)
    classifier_robot2.evaluate_result(robot2_test_y, y_predict_robot2)


    classifier_iris = KNN()
    classifier_iris.fit(Iris_train_X, Iris_train_y, 2)
    y_predict_iris = classifier_iris.predict(Iris_test_X)
    print(Iris_test_y.shape, y_predict_iris.shape)
    classifier_iris.evaluate_result(Iris_test_y, y_predict_iris)



    #########################################


    robot1 = pd.read_csv('RobotDataset/Robot1',header=None, delim_whitespace=True)
    robot2 = pd.read_csv('RobotDataset/Robot2',header=None, delim_whitespace=True)
    robot1.columns = robot2.columns = ['class', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'id']
    iris_validation = pd.read_csv(sys.argv[3], header=None)

    robot1_validation.columns = robot1.columns
    robot2_validation.columns = robot2.columns
    iris_validation.columns = iris_data.columns

    robot1_validation = robot1_validation.drop('id', 1)
    robot2_validation = robot2_validation.drop('id', 1)
    iris_validation = iris_data.drop('class', 1)

    robot1_result = classifier_robot1.predict(robot1_validation)
    robot2_result = classifier_robot2.predict(robot2_validation)
    iris_result = classifier_iris.predict(iris_validation)
    iris_result['predicted'] = [ convert_back_Iris_to_categorical[x] for x in iris_result['predicted']]
 
    robot1_result.to_csv("robot1_result.csv")
    robot2_result.to_csv("robot2_result.csv")
    iris_result.to_csv("iris_result.csv")