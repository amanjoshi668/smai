import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
import math
from IPython.core.display import display, HTML


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
    pprint(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("tn = {0}, fp = {1}, fn = {2}, tp = {3}".format(tn, fp, fn, tp))
    f1 = f1_score(y_test, y_pred, average="macro")
    print("f1_score\t : ", f1)
    ps = precision_score(y_test, y_pred, average="macro")
    print("precision_score\t : ", ps)
    rs = recall_score(y_test, y_pred, average="macro")
    print("recall_score\t : ", rs)
    acs = accuracy_score(y_test, y_pred)
    print("accuracy_score\t : ", acs)
    return np.array([acs, ps, rs, f1])


class NaiveBayes():
    def calculate_count(self, df):
        labels, counts = np.unique(df, return_counts=True)
        return zip(labels, counts)

    def seperate_by_class(self):
        self.continuous_summary = pd.DataFrame(
            columns=self.continuous_features)
        self.categorical_summary = pd.DataFrame(
            columns=self.categorical_features)
        distinct_classes = np.unique(self.train_y)
        self.train_X['class'] = self.train_y
        self.class_frequency = {}
        for label in distinct_classes:
            class_data = self.train_X.where(
                self.train_X['class'] == label).dropna()
            self.continuous_summary.loc[label] = [{
                'mean':
                class_data[column].mean(),
                'std_dev':
                class_data[column].std()
            } for column in self.continuous_features]
            self.class_frequency[label] = len(class_data) / len(self.train_X)
            #             print(np.unique(class_data['credit_card'], return_counts=True))
            self.categorical_summary.loc[label] = [{
                value: count / len(class_data)
                for (value, count) in self.calculate_count(class_data[column])
            } for column in categorical_features]
        self.train_X = self.train_X.drop('class', 1)
        display(self.continuous_summary)
        display(self.categorical_summary)

    def calculate_probability(self, x, mean, std_dev):
        exponent = math.exp((-(x - mean)**2) / (2 * (std_dev**2)))
        return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent

    def train(self, train_X, train_y, categorical_features,
              continuous_features):
        self.train_X = train_X
        self.train_y = train_y
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.seperate_by_class()

    def predict_row(self, query):
        continuous_probability = []
        for index, row in self.continuous_summary.iterrows():
            continuous_probability.append(
                np.prod([
                    self.calculate_probability(query[key], row[key]['mean'],
                                               row[key]['std_dev'])
                    for key in self.continuous_features
                ]) * self.class_frequency[index])
        categorical_probability = []
        for index, row in self.categorical_summary.iterrows():
            categorical_probability.append(
                np.prod([
                    row[key][query[key]] for key in self.categorical_features
                ]) * self.class_frequency[index])
        probability = [
            i * j
            for i, j in zip(continuous_probability, categorical_probability)
        ]
        return list(
            self.categorical_summary.index.values)[np.argmax(probability)]

    def predict(self, test_X):
        test_y = pd.DataFrame()
        queries = test_X.to_dict(orient="records")
        test_y['predicted'] = [self.predict_row(row) for row in queries]
        return test_y


if __name__ == "__main__":
    Bank_data = pd.read_csv('LoanDataset/data.csv')
    categorical_features = [
        'education_level',
        'certificate_of_deposit',
        'internet_banking',
        'credit_card',  # ,'zip'
    ]
    label = 'class'
    continuous_features = [
        'age', 'no_of_year_of_exp', 'annual_income', 'family_size',
        'average_spending', 'motgage_value', 'security_account'
    ]
    Bank_data = Bank_data.where(Bank_data['no_of_year_of_exp'] >= 0).dropna()
    one_data = Bank_data.where(Bank_data['class'] == 1).dropna()
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.append(one_data, ignore_index=True)
    Bank_data = Bank_data.reset_index(drop=True)
    Bank_y = Bank_data[[label]]
    Bank_X = Bank_data.drop(label, 1)
    # print(X.shape, y.shape, data.shape)
    Bank_train_X, Bank_test_X, Bank_train_y, Bank_test_y = split_test_train(
        Bank_X, Bank_y, 0.8)

    NB = NaiveBayes()
    NB.train(Bank_train_X, Bank_train_y,
             categorical_features, continuous_features)

    
    Bank_validation_X = pd.read_csv(sys.argv[1], header=True)
    # print(Bank_validation_X.values)
    Bank_validation_X.columns = Bank_data.columns
    # print(Bank_validation_X.columns)
    # Bank_validation_X = Bank_validation_X.drop('class', 1)
    print(Bank_test_X.head())
    print(Bank_validation_X.head())
    predicted_bank = NB.predict(Bank_test_X)
    
    result = evaluate_result(Bank_test_y, predicted_bank)

    predicted_bank = NB.predict(Bank_validation_X)
    predicted_bank.to_csv('predicted_bank.csv')
    print(result)