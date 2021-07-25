# -*- coding: utf-8 -*-
#
# Copyright 2020 Pietro Barbiero, Alberto Tonda and Giovanni Squillero
# Licensed under the EUPL

import unittest


class TestPFE(unittest.TestCase):

    def test_PFE(self):

        from sklearn.datasets import make_friedman1
        from sklearn.feature_selection import RFE
        from sklearn.svm import SVR
        from pfe.pfe import PFE

        X, y = make_friedman1(n_samples=50, n_features=1000, random_state=0)
        estimator = SVR(kernel="linear")

        selector = RFE(estimator, 5, step=1)
        selector = selector.fit(X, y)
        print(sum(selector.support_))

        selector = PFE(estimator)
        selector = selector.fit(X, y)
        print(sum(selector.support_))

    def test_dataset(self):
        from sklearn.datasets import load_wine
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import RidgeClassifier
        from sklearn import clone
        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import Pipeline
        from pfe.pfe import PFE

        seed = 42

        X, y = load_wine(return_X_y=True)

        clf = RidgeClassifier(random_state=seed)

        pipeline_rfe = Pipeline([("rfe", RFE(clone(clf))), ("clf", clone(clf))])
        pipeline_pfe = Pipeline([("pfe", PFE(clone(clf))), ("clf", clone(clf))])

        pipeline_rfe.fit(X, y)
        score_rfe = pipeline_rfe.score(X, y)
        n_features_rfe = sum(pipeline_rfe.steps[0][1].support_)

        pipeline_pfe.fit(X, y)
        score_pfe = pipeline_pfe.score(X, y)
        n_features_pfe = sum(pipeline_pfe.steps[0][1].support_)

        print()
        print("RFE score: %.4f; #features: %d" % (score_rfe, n_features_rfe))
        print("PFE score: %.4f; #features: %d" % (score_pfe, n_features_pfe))

    def test_profile(self):

        from sklearn.datasets import make_friedman1
        from sklearn.svm import SVR
        from pfe.pfe import PFE
        import cProfile

        X, y = make_friedman1(n_samples=50, n_features=1000, random_state=0)
        estimator = SVR(kernel="linear")

        selector = PFE(estimator)
        selector.fit(X, y)

        cProfile.runctx('selector.fit(X, y)', None, locals())

    # def test_benchmark_classification(self):
    #
    #     from sklearn.feature_selection import RFE
    #     from sklearn.linear_model import RidgeClassifier
    #     from sklearn import clone
    #     from lazygrid.datasets import load_openml_dataset
    #     from sklearn.model_selection import cross_validate
    #     from sklearn.pipeline import Pipeline
    #     from pfe.pfe import PFE
    #     import pandas as pd
    #     import scipy
    #     import os
    #
    #     seed = 42
    #     cv = 10
    #
    #     X, y, n_classes = load_openml_dataset(dataset_name="leukemia")
    #
    #     clf = RidgeClassifier(random_state=seed)
    #
    #     pipeline_rfe = Pipeline([("rfe", RFE(clone(clf))), ("clf", clone(clf))])
    #     pipeline_pfe = Pipeline([("pfe", PFE(clone(clf))), ("clf", clone(clf))])
    #
    #     # cross validation
    #     scores_rfe = cross_validate(pipeline_rfe, X, y, n_jobs=-1, cv=cv, return_train_score=True, return_estimator=True)
    #     scores_pfe = cross_validate(pipeline_pfe, X, y, cv=cv, return_train_score=True, return_estimator=True)
    #
    #     scores_pfe["avg_fit_time"] = scipy.average(scores_pfe["fit_time"])
    #     scores_pfe["sem_fit_time"] = scipy.stats.sem(scores_pfe["fit_time"])
    #     scores_pfe["avg_train_score"] = scipy.average(scores_pfe["train_score"])
    #     scores_pfe["sem_train_score"] = scipy.stats.sem(scores_pfe["train_score"])
    #     scores_pfe["avg_test_score"] = scipy.average(scores_pfe["test_score"])
    #     scores_pfe["sem_test_score"] = scipy.stats.sem(scores_pfe["test_score"])
    #     n_features_list = [sum(estimator.steps[0][1].support_) for estimator in scores_pfe["estimator"]]
    #     scores_pfe["n_features_selected"] = n_features_list
    #     scores_pfe["avg_n_features"] = scipy.average(n_features_list)
    #     scores_pfe["sem_n_features"] = scipy.stats.sem(n_features_list)
    #
    #     scores_rfe["avg_fit_time"] = scipy.average(scores_rfe["fit_time"])
    #     scores_rfe["sem_fit_time"] = scipy.stats.sem(scores_rfe["fit_time"])
    #     scores_rfe["avg_train_score"] = scipy.average(scores_rfe["train_score"])
    #     scores_rfe["sem_train_score"] = scipy.stats.sem(scores_rfe["train_score"])
    #     scores_rfe["avg_test_score"] = scipy.average(scores_rfe["test_score"])
    #     scores_rfe["sem_test_score"] = scipy.stats.sem(scores_rfe["test_score"])
    #     n_features_list = [sum(estimator.steps[0][1].support_) for estimator in scores_rfe["estimator"]]
    #     scores_rfe["n_features_selected"] = n_features_list
    #     scores_rfe["avg_n_features"] = scipy.average(n_features_list)
    #     scores_rfe["sem_n_features"] = scipy.stats.sem(n_features_list)
    #
    #     scores = pd.DataFrame()
    #     scores = pd.concat([scores, pd.DataFrame.from_records(scores_pfe)], ignore_index=True)
    #     scores = pd.concat([scores, pd.DataFrame.from_records(scores_rfe)], ignore_index=True)
    #
    #     results_dir = "./results"
    #     if not os.path.isdir(results_dir):
    #         os.makedirs(results_dir)
    #     scores.to_csv(os.path.join(results_dir, "benchmark_classification.csv"))
    #
    #     print()


suite = unittest.TestLoader().loadTestsFromTestCase(TestPFE)
unittest.TextTestRunner(verbosity=2).run(suite)
