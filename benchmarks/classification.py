from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, Ridge
from sklearn import clone
from lazygrid.datasets import load_openml_dataset
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dfe import DFE
from pfe.pfe import PFE
import pandas as pd
import scipy
import os

seed = 42
cv = 10
results_dir = "./test3"
global_scores = pd.DataFrame()

datasets = [
    # "ionosphere",   # 351 instances     35 features     2 classes
    # "madelon",      # 2600 instances    500 features    2 classes
    "isolet",       # 1560 instances    617 features    26 classes

    # "gisette",      # 7000 instances    5000 features   2 classes
    # "leukemia",     # 72 instances      7130 features   2 classes
    # "arcene",       # 200 instances     10000 features  2 classes
]

for dataset in datasets:

    X, y, n_classes = load_openml_dataset(dataset_name=dataset)

    clf = RidgeClassifier(random_state=seed)
    logc = RidgeClassifier(random_state=seed)
    logr = Ridge(random_state=seed)
    # regr = RandomForestRegressor(random_state=seed)

    pipeline_dfe = Pipeline([("dfe", DFE(clone(clf), base_score=1, verbose=1)), ("clf", clone(clf))])
    pipeline_pfe = Pipeline([("pfe", PFE(clone(clf), clone(logr), clone(logc), verbose=1)), ("clf", clone(clf))])
    pipeline_rfe = Pipeline([("rfe", RFE(clone(clf), verbose=1)), ("clf", clone(clf))])

    # cross validation
    scores_dfe = cross_validate(pipeline_dfe, X, y, n_jobs=-1, cv=cv, return_train_score=True, return_estimator=True)
    # scores_pfe = cross_validate(pipeline_pfe, X, y, n_jobs=1, cv=cv, return_train_score=True, return_estimator=True)
    scores_rfe = cross_validate(pipeline_rfe, X, y, n_jobs=-1, cv=cv, return_train_score=True, return_estimator=True)

    n_features_dfe = [sum(estimator.steps[0][1].support_) for estimator in scores_dfe["estimator"]]
    scores_dfe["n_features_selected"] = n_features_dfe
    # n_features_pfe = [sum(estimator.steps[0][1].support_) for estimator in scores_pfe["estimator"]]
    # scores_pfe["n_features_selected"] = n_features_pfe
    n_features_rfe = [sum(estimator.steps[0][1].support_) for estimator in scores_rfe["estimator"]]
    scores_rfe["n_features_selected"] = n_features_rfe

    scores = pd.DataFrame()
    scores = pd.concat([scores, pd.DataFrame.from_records(scores_dfe)], ignore_index=True)
    # scores = pd.concat([scores, pd.DataFrame.from_records(scores_pfe)], ignore_index=True)
    scores = pd.concat([scores, pd.DataFrame.from_records(scores_rfe)], ignore_index=True)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    scores.to_csv(os.path.join(results_dir, dataset + ".csv"))

    global_scores_dfe = {
        "dataset": dataset,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "method": "DFE",
        "avg_fit_time": scipy.average(scores_dfe["fit_time"]),
        "sem_fit_time": scipy.stats.sem(scores_dfe["fit_time"]),
        "avg_train_score": scipy.average(scores_dfe["train_score"]),
        "sem_train_score": scipy.stats.sem(scores_dfe["train_score"]),
        "avg_test_score": scipy.average(scores_dfe["test_score"]),
        "sem_test_score": scipy.stats.sem(scores_dfe["test_score"]),
        "avg_n_features": scipy.average(n_features_dfe),
        "sem_n_features": scipy.stats.sem(n_features_dfe)
    }

    # global_scores_pfe = {
    #     "dataset": dataset,
    #     "n_samples": X.shape[0],
    #     "n_features": X.shape[1],
    #     "n_classes": n_classes,
    #     "method": "PFE",
    #     "avg_fit_time": scipy.average(scores_pfe["fit_time"]),
    #     "sem_fit_time": scipy.stats.sem(scores_pfe["fit_time"]),
    #     "avg_train_score": scipy.average(scores_pfe["train_score"]),
    #     "sem_train_score": scipy.stats.sem(scores_pfe["train_score"]),
    #     "avg_test_score": scipy.average(scores_pfe["test_score"]),
    #     "sem_test_score": scipy.stats.sem(scores_pfe["test_score"]),
    #     "avg_n_features": scipy.average(n_features_pfe),
    #     "sem_n_features": scipy.stats.sem(n_features_pfe)
    # }

    global_scores_rfe = {
        "dataset": dataset,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "method": "RFE",
        "avg_fit_time": scipy.average(scores_rfe["fit_time"]),
        "sem_fit_time": scipy.stats.sem(scores_rfe["fit_time"]),
        "avg_train_score": scipy.average(scores_rfe["train_score"]),
        "sem_train_score": scipy.stats.sem(scores_rfe["train_score"]),
        "avg_test_score": scipy.average(scores_rfe["test_score"]),
        "sem_test_score": scipy.stats.sem(scores_rfe["test_score"]),
        "avg_n_features": scipy.average(n_features_rfe),
        "sem_n_features": scipy.stats.sem(n_features_rfe)
    }

    global_scores = pd.concat([global_scores, pd.DataFrame.from_records(global_scores_dfe, index=["dataset"])], ignore_index=True)
    # global_scores = pd.concat([global_scores, pd.DataFrame.from_records(global_scores_pfe, index=["dataset"])], ignore_index=True)
    global_scores = pd.concat([global_scores, pd.DataFrame.from_records(global_scores_rfe, index=["dataset"])], ignore_index=True)

    global_scores.to_csv(os.path.join(results_dir, "summary.csv"))
