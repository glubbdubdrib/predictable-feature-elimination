# -*- coding: utf-8 -*-
#
# Copyright 2020 Pietro Barbiero, Alberto Tonda and Giovanni Squillero
# Licensed under the EUPL

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import check_X_y, safe_sqr
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


class DPFE(RFE):

    def __init__(self, estimator, base_score=0.9, n_splits=10, random_state=42,
                 n_features_to_select=None, verbose=0):
        super().__init__(estimator, n_features_to_select, verbose)
        self.verbose = verbose
        self.base_score = base_score
        self.n_splits = n_splits
        self.random_state = random_state

    def _fit(self, X, y, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")
        X = pd.DataFrame(X)

        n_samples, n_features = X.shape
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        worst_feature = 0

        # Recursive elimination
        i = 1
        while np.sum(support_) > n_features_to_select:

            if worst_feature == n_features:
                break

            z = (np.cov(X.T) ** 2)[:, support_].sum(axis=1)
            z[support_==False] = 0
            z_max = np.argmax(z)
            z_2 = z
            z_2[z_max] = 0
            z_2 = z_2 / np.max(z_2)
            a = z_2 > 0.5

            support_[z_max] = False
            X_worse = X.iloc[:, z_max]
            X_reduced = X.iloc[:, a]

            skf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            train_index, val_index = [split for split in skf.split(X_worse)][0]
            X_train, X_val = X_reduced.iloc[train_index], X_reduced.iloc[val_index]
            y_train, y_val = X_worse[train_index], X_worse[val_index]

            # Eliminate predictable features
            if self.verbose > 0:
                print("Fitting estimator with %d features (%d/%d)" % (np.sum(support_), i, n_features))
                i += 1

            # estimator = clone(self.estimator)
            # estimator.fit(X_train, y_train)
            # score = estimator.score(X_val, y_val)
            # define model
            input_shape = X_train.shape[1]
            model = Sequential()
            model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(input_shape,)))
            model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
            model.add(Dense(1))
            # compile the model
            model.compile(optimizer='adam', loss='mse')
            # fit the model
            model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
            # evaluate the model
            yhat = model.predict(X_val)
            score = r2_score(y_val, yhat)

            if score >= self.base_score:

                # Compute step score on the previous selection iteration
                # because 'estimator' must use features
                # that have not been eliminated yet
                ranking_[np.logical_not(support_)] += 1

            else:
                support_[z_max] = True

            worst_feature += 1

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


def _is_integer(x):
    return np.equal(np.mod(x, 1), 0)



# from sklearn.datasets import load_iris
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Dense
# import numpy as np
#
#
# X, y = load_iris(return_X_y=True)
#
# z = (np.cov(X.T)**2).sum(axis=1)
# z_max = np.argmax(z)
# z_2 = z
# z_2[z_max] = 0
# z_2 = z_2 / np.max(z_2)
# a = z_2 > 0.5
#
# x = X[:, a]
# y = X[:, z_max]
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
# # determine the number of input features
# n_features = X_train.shape[1]
# # define model
# model = Sequential()
# model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1))
# # compile the model
# model.compile(optimizer='adam', loss='mse')
# # fit the model
# model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# # evaluate the model
# yhat = model.predict(X_test)
# r2_score(y_test, yhat)
