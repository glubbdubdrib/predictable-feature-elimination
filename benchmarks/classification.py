import sys

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
    # "iris",
    # "yeast_ml8",                            # 2417 samples      116 features        2 classes
    # "scene",                                # 2407 samples      299 features        2 classes
    # "madelon",                              # 2600 samples      500 features        2 classes
    # "isolet",                               # 7797 samples      617 features        26 classes
    # "gina_agnostic",                        # 3468 samples      970 features        2 classes
    "gisette",                              # 7000 samples      5000 features       2 classes
    "amazon-commerce-reviews",              # 1500 samples      10000 features      50 classes
    "OVA_Colon",                            # 1545 samples      10936 features      2 classes
    "GCM",                                  # 190 samples       16063 features      14 classes
    "Dexter",                               # 600 samples       20000 features      2 classes
    "variousCancers_final",                 # 383 samples       54676 features      9 classes
    "anthracyclineTaxaneChemotherapy",      # 159 samples       61360 features      2 classes
    "Dorothea",                             # 1150 samples      100000 features     2 classes
]


def main():

    # Cross-validation params
    cv = 10
    n_jobs = -1
    seed = 42
    results_dir = "./results"

    overall_scores = pd.DataFrame()

    for dataset in datasets:

        verbose_scores = pd.DataFrame()
        summary_scores = pd.DataFrame()

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        X, y, n_classes = load_openml_dataset(dataset_name=dataset)

        clf = RidgeClassifier(random_state=seed)
        regr = Ridge(random_state=seed)
        est = RidgeClassifier(random_state=seed)

        sc = StandardScaler()
        pipelines = [
            Pipeline([("sc", sc), ("fs", DFE(clone(regr), base_score=0.9, verbose=1)), ("est", clone(est))]),
            Pipeline([("sc", sc), ("fs", RFE(clone(clf), verbose=1)), ("est", clone(est))]),
            # Pipeline([("pfe", PFE(clone(clf), clone(logr), clone(logc), verbose=1)), ("clf", clone(clf))]),
        ]

        for pipeline in pipelines:

            # cross validation
            scores = cross_validate(pipeline, X, y, n_jobs=n_jobs, cv=cv,
                                    return_train_score=True, return_estimator=True)

            n_features = [sum(estimator.steps[1][1].support_) for estimator in scores["estimator"]]
            scores["n_features_selected"] = n_features

            scores = pd.DataFrame.from_records(scores)
            verbose_scores = pd.concat([verbose_scores, scores], ignore_index=True)
            verbose_scores.to_csv(os.path.join(results_dir, dataset + "_verbose.csv"))

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
            summary_scores = pd.concat([summary_scores, summary], ignore_index=True)
            summary_scores.to_csv(os.path.join(results_dir, dataset + "_summary.csv"))

            overall_scores = pd.concat([overall_scores, summary], ignore_index=True)
            overall_scores.to_csv(os.path.join(results_dir, "overall_results.csv"))


if __name__ == "__main__":
    sys.exit(main())
