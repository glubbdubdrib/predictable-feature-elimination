# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero, Alberto Tonda and Giovanni Squillero
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import check_X_y, safe_sqr


class DFE(RFE):

    def __init__(self, estimator, base_score=0.9, min_corr=0.5, n_splits=10, random_state=42,
                 n_features_to_select=None, verbose=0):
        super().__init__(estimator, n_features_to_select, verbose)
        self.verbose = verbose
        self.base_score = base_score
        self.n_splits = n_splits
        self.random_state = random_state
        self.min_corr = min_corr

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

        # compute correlation matrix
        # and sort feature by highest mean squared correlation
        C = np.square(np.corrcoef(X.T) - np.diag(np.ones(X.shape[1])))
        coefs = C.mean(axis=1)

        # Get ranks
        ranks = np.argsort(-safe_sqr(coefs))
        worst_feature = 0

        # Recursive elimination
        i = 1
        while np.sum(support_) > n_features_to_select:

            if worst_feature == n_features:
                break

            support_[ranks[worst_feature]] = False
            X_worse = X.iloc[:, ranks[worst_feature]]

            correlation_to_worst_feature = -C[:, ranks[worst_feature]]
            correlation_to_worst_feature[support_ == False] = 0
            most_related_features = np.argsort(correlation_to_worst_feature)
            sorted_support = support_[most_related_features]
            if self.min_corr < np.max(-correlation_to_worst_feature):
                sorted_support = sorted_support & (-correlation_to_worst_feature[most_related_features] > self.min_corr)

            X_reduced = X.iloc[:, most_related_features[sorted_support]]

            skf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            train_index, val_index = [split for split in skf.split(X_worse)][0]
            X_train, X_val = X_reduced.iloc[train_index], X_reduced.iloc[val_index]
            y_train, y_val = X_worse[train_index], X_worse[val_index]

            # Eliminate predictable features
            if self.verbose > 0:
                print("Fitting estimator with %d features (%d/%d)" % (np.sum(support_), i, n_features))
                i += 1

            estimator = clone(self.estimator)
            estimator.fit(X_train, y_train)
            score = estimator.score(X_val, y_val)

            if score >= self.base_score:

                # Compute step score on the previous selection iteration
                # because 'estimator' must use features
                # that have not been eliminated yet
                ranking_[np.logical_not(support_)] += 1

            else:
                support_[ranks[worst_feature]] = True

            worst_feature += 1

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


def _is_integer(x):
    return np.equal(np.mod(x, 1), 0)
