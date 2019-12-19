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
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import check_X_y, safe_sqr


class PFE(RFE):

    def __init__(self, estimator, regression_estimator=LinearRegression(),
                 classification_estimator=LogisticRegression(), verbose=0):
        super().__init__(estimator=estimator, n_features_to_select=1, step=1, verbose=verbose)
        self.regression_estimator = regression_estimator
        self.classification_estimator = classification_estimator

    def _fit(self, X, y, step_score=None, base_score=0.9):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")

        n_samples, n_features = X.shape

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Remaining features
        features = np.arange(n_features)[support_]

        # Rank the remaining features
        estimator = clone(self.estimator)

        if self.verbose > 0:
            print("Fitting estimator with %d features." % n_features)
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
        if coefs.ndim > 1:
            ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
        else:
            ranks = np.argsort(safe_sqr(coefs))

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)

        worst_feature = 0
        if n_samples < np.sum(support_):
            # Eliminate collinear features
            threshold = np.sum(support_) - n_samples

            # Find worst feature
            worst_feature = threshold + 1

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold + 1]] = False
            ranking_[np.logical_not(support_)] += 1

        X_worse = X[:, worst_feature]

        # Recursive elimination
        while True:

            # Remaining features
            features = np.arange(n_features)[support_]

            if np.all(_is_integer(X_worse)):
                # Classification problem
                estimator = clone(self.classification_estimator)
            else:
                # Regression problem
                estimator = clone(self.regression_estimator)

            # Eliminate predictable features
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], X_worse)
            score = estimator.score(X[:, features], X_worse)

            worst_feature += 1

            if score >= base_score:

                # Compute step score on the previous selection iteration
                # because 'estimator' must use features
                # that have not been eliminated yet
                if step_score:
                    self.scores_.append(step_score(estimator, features))
                support_[ranks[worst_feature]] = False
                ranking_[np.logical_not(support_)] += 1

            # Find the worst feature
            X_worse = X[:, ranks[worst_feature]]

            if n_features - worst_feature < 2:
                break

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
