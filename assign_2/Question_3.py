import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from heapq import heappush, heappop
import sys


def mean_square(X, y, theta):
    return np.sum(np.power(((X @ theta.T) - y), 2)) / (2 * len(X))


def mean_absolute(X, y, theta):
    #     print("here")
    return np.sum(abs(X @ theta.T) - y) / (len(X))


def mean_absolute_percentage(X, y, theta):
    return np.sum(abs(((X @ theta.T) - y) / y)) * 100 / len(X)

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

class LinearRegression:
    def __init__(self, alpha=0.01, iterariton=5000, cost_function=mean_square):
        self.alpha = alpha
        self.iterariton = iterariton
        self.cost_function = cost_function

    def standardize(self, X):
        X_standardized = (X - self.mean) / self.var
        return X_standardized

    def gradient_Descent(self, train_X, train_y):
        self.cost = np.zeros(self.iterariton)
        for i in range(self.iterariton):
            if self.cost_function.__name__ == "mean_square":
                self.theta = self.theta - (self.alpha / len(train_X)) * np.sum(
                    train_X * (train_X @ self.theta.T - train_y), axis=0)
            elif self.cost_function.__name__ == "mean_absolute":
                loss = (train_X @ self.theta.T) - train_y
                #                 loss = [1 if x >= 0 else -1 for x in loss]|
                loss[loss >= 0] = 1
                loss[loss < 0] = -1
                gradient = np.sum(
                    (train_X * loss), axis=0) / len(train_X)  # shape : (n,1)
                self.theta = self.theta - self.alpha * (gradient)
#                 self.theta = self.theta - (
#                     self.alpha / len(self.train_X)) * np.sum(
#                         self.train_X , loss)
#                 loss = h - Y ## shape: (m,1)
#                 loss[loss >= 0] = 1
#                 loss[loss < 0] = -1
#                 gradient = np.dot(X.T,loss) / m  ##shape : (n,1)
#                 theta = theta - alpha * (gradient.T)

            elif (self.cost_function.__name__ == "mean_absolute_percentage"):
                loss = train_X @ self.theta.T - train_y
                #                 loss = [1 if x >= 0 else -1 for x in loss]|
                loss[loss >= 0] = 1
                loss[loss < 0] = -1
                gradient = np.dot((train_X / np.abs(train_y)).T, loss) / len(
                    train_X)  # shape : (n,1)
                self.theta = self.theta - self.alpha * (gradient.T)
            self.cost[i] = self.cost_function(train_X, train_y, self.theta)


#             print(cost[i])


    def train(self, train_X, train_y):
        self.mean, self.var = train_X.mean(), train_X.std()
        train_X = self.standardize(train_X)
        train_X = train_X.values
        ones = np.ones([train_X.shape[0], 1])
        train_X = np.concatenate((ones, train_X), axis=1)
        train_y = train_y.values
        self.theta = np.zeros([1, train_X.shape[1]])
        self.gradient_Descent(train_X, train_y)

    def predict(self, test_X):
        test_X = self.standardize(test_X)
        test_X = test_X.values
        ones = np.ones([test_X.shape[0], 1])
        test_X = np.concatenate((ones, test_X), axis=1)
        return (test_X @ self.theta.T).flatten()


if __name__ == "__main__":
    Admission_data = pd.read_csv('AdmissionDataset/data.csv')
    Admission_X = Admission_data.drop(
        ['Serial No.', 'Chance of Admit '], axis=1)
    Admission_y = Admission_data[['Chance of Admit ']]
    Admission_train_X, Admission_test_X, Admission_train_y, Admission_test_y = split_test_train(
        Admission_X, Admission_y)
    # print(Admission_test_X.shape)
    cost_function = [mean_absolute, mean_absolute_percentage, mean_square]
    for function in cost_function:
        LR = LinearRegression(cost_function=function)
        LR.train(Admission_train_X, Admission_train_y)
        predicted_chance = LR.predict(Admission_test_X)
        print(function.__name__, r2_score(Admission_test_y, predicted_chance))

    LR = LinearRegression()
    LR.train(Admission_train_X, Admission_train_y)
    Admission_validation_data = pd.read_csv(sys.argv[1], header=None)
    Admission_validation_data.columns = Admission_data.values
    Admission_validation_data = Admission_validation_data.drop(
        ['Serial No.', 'Chance of Admit '], axis=1)
    # Admission_validation_data.columns = Admission_data.columns
    predicted_chance = LR.predict(Admission_validation_data)
    df = pd.DataFrame()
    df['Chance of Admit '] = predicted_chance
    df.to_csv("Admission_result.csv")
