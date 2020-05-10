from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, Ridge
from sklearn import clone
from lazygrid.datasets import load_openml_dataset
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dfe import DFE
from pfe.pfe import PFE
import pandas as pd
import scipy
import os

datasets = [
    # "diabetes",
    # "triazines",
    # "tecator",
    "mtp2",
]

# Cross-validation params
cv = 3
n_jobs = 1
seed = 42
results_dir = "./bench3"

global_scores = pd.DataFrame()

for dataset in datasets:

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    X, y, n_classes = load_openml_dataset(dataset_name=dataset)

    clf = LogisticRegression(random_state=seed)
    regr = LinearRegression()
    est = SVR()

    sc = StandardScaler()
    pipelines = [
        Pipeline([("sc", sc), ("fs", DFE(clone(regr), base_score=0.98, verbose=1)), ("est", clone(est))]),
        Pipeline([("sc", sc), ("fs", RFE(clone(regr), verbose=1)), ("est", clone(est))]),
        # Pipeline([("pfe", PFE(clone(clf), clone(logr), clone(logc), verbose=1)), ("clf", clone(clf))]),
    ]

    for pipeline in pipelines:

        # cross validation
        scores = cross_validate(pipeline, X, y, n_jobs=n_jobs, cv=cv,
                                return_train_score=True, return_estimator=True)

        n_features = [sum(estimator.steps[1][1].support_) for estimator in scores["estimator"]]
        scores["n_features_selected"] = n_features

        scores = pd.DataFrame.from_records(scores)
        scores.to_csv(os.path.join(results_dir, dataset + ".csv"))

        summary = {
            "dataset": dataset,
            "samples": X.shape[0],
            "features": X.shape[1],
            "classes": n_classes,
            "method": pipeline.steps[1][1].__class__.__name__,
            "avg fit time": scipy.average(scores["fit_time"]),
            "sem fit time": scipy.stats.sem(scores["fit_time"]),
            "avg train score": scipy.average(scores["train_score"]),
            "sem train score": scipy.stats.sem(scores["train_score"]),
            "avg test score": scipy.average(scores["test_score"]),
            "sem test score": scipy.stats.sem(scores["test_score"]),
            "avg n features": scipy.average(n_features),
            "sem n features": scipy.stats.sem(n_features)
        }

        summary = pd.DataFrame.from_records(summary, index=["dataset"])
        global_scores = pd.concat([global_scores, summary], ignore_index=True)
        global_scores.to_csv(os.path.join(results_dir, "summary.csv"))
