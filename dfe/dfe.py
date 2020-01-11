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

        estimator = clone(self.estimator)
        estimator.fit(X, y)

        # Get coefs
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" or "feature_importances_" '
                               'attributes')

        # Get ranks
        ranks = np.argsort(safe_sqr(coefs))
        worst_feature = 0

        # Recursive elimination
        i = 1
        while np.sum(support_) > n_features_to_select:

            if worst_feature == n_features:
                break

            support_[ranks[worst_feature]] = False
            X_worse = X.iloc[:, ranks[worst_feature]]

            skf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            train_index, val_index = [split for split in skf.split(X_worse)][0]
            X_train, X_val = X.iloc[train_index, support_], X.iloc[val_index, support_]
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

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X.iloc[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


def _is_integer(x):
    return np.equal(np.mod(x, 1), 0)
