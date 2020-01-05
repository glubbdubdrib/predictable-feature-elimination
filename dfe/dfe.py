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
from sklearn import clone
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import check_X_y, safe_sqr


class DFE(RFE):

    def __init__(self, estimator, base_score=1.0, n_features_to_select=None, verbose=0):
        super().__init__(estimator, n_features_to_select, verbose)
        self.verbose = verbose
        self.base_score = base_score

    def _fit(self, X, y, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")

        n_samples, n_features = X.shape
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Remaining features
        features = np.arange(n_features)[support_]

        if self.verbose > 0:
            print("Computing MI with %d features." % n_features)

        skb = SelectKBest(score_func=mutual_info_classif, k=1)
        skb.fit(X, y)
        coefs = skb.scores_

        # Get ranks
        ranks = np.argsort(safe_sqr(coefs))

        worst_feature = 0
        if n_samples < np.sum(support_):
            # Number of collinear features
            n_collinear = np.sum(support_) - n_samples

            # Eliminate features up to the threshold
            threshold = np.min([n_features_to_select, n_collinear])

            # Find worst feature
            worst_feature = threshold + 1

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            support_[features[ranks][:threshold + 1]] = False
            ranking_[np.logical_not(support_)] += 1

        X_worse = X[:, worst_feature]

        # Recursive elimination
        i = 0
        while np.sum(support_) > n_features_to_select:

            if worst_feature == n_features - 1:
                break

            # Remaining features
            features = np.arange(n_features)[support_]

            # Eliminate predictable features
            if self.verbose > 0:
                print("Iteration %d: worst: %d" % (i, worst_feature))
                i += 1
                print("Selected features: %d." % np.sum(support_))

            # Classification problem
            try:
                skb = SelectKBest(score_func=mutual_info_classif, k=1)
                skb.fit(X[:, features], X_worse)
                score = skb.scores_
            except ValueError:
                skb = SelectKBest(score_func=mutual_info_regression, k=1)
                skb.fit(X[:, features], X_worse)
                score = skb.scores_

            # score /= np.max(score)
            score = np.max(score)

            worst_feature += 1

            if score >= self.base_score:

                # Compute step score on the previous selection iteration
                # because 'estimator' must use features
                # that have not been eliminated yet
                support_[ranks[worst_feature]] = False
                ranking_[np.logical_not(support_)] += 1

            # Find the worst feature
            X_worse = X[:, ranks[worst_feature]]

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


def _is_integer(x):
    return np.equal(np.mod(x, 1), 0)
