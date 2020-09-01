import sys
import scipy
import os

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn import clone
from lazygrid.datasets import load_openml_dataset
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from dfe.dfe import DFE
from skfwrapper.skfwrapper import SKF_lap, SKF_mcfs, SKF_spec, SKF_ndfs, SKF_udfs

random_state = 42
X1, y1 = make_regression(n_samples=2600, n_features=500,
                         n_informative=100, effective_rank=None,
                         random_state=random_state)
X2, y2 = make_regression(n_samples=2600, n_features=500,
                         n_informative=200, effective_rank=50,
                         random_state=random_state)
X3, y3 = make_regression(n_samples=2600, n_features=500,
                         n_informative=300, effective_rank=150,
                         random_state=random_state)

datasets = [
    # "iris",

    ["genreg-100", X1, y2],
    ["genreg-300", X1, y2],
    ["genreg-500", X1, y2],

    # "yeast_ml8",                    # 2417 samples      116 features    2 classes
    # "scene",                        # 2407 samples      299 features    2 classes
    # "isolet",                       # 7797 samples      617 features    26 classes
    # "gina_agnostic",                # 3468 samples      970 features    2 classes
    # "gas-drift",                    # 13910 samples     129 features    6 classes
    # "letter",                       # 20000 samples     17 samples      26 classes

    # "mozilla4",                     # 15545 samples     6 features      2 classes
    # "Amazon_employee_access",       # 32769 samples     10 features     2 classes
    # "electricity",                  # 45312 samples     9 features      2 classes
    # "mnist_784",                    # 70000 samples     785 features    10 classes
    # "covertype",                    # 581012 samples    55 features     7 classes

    # "gisette",                              # 7000 samples      5000 features       2 classes
    # "amazon-commerce-reviews",              # 1500 samples      10000 features      50 classes
    # "OVA_Colon",                            # 1545 samples      10936 features      2 classes
    # "GCM",                                  # 190 samples       16063 features      14 classes
    # "Dexter",                               # 600 samples       20000 features      2 classes
    # "variousCancers_final",                 # 383 samples       54676 features      9 classes
    # "anthracyclineTaxaneChemotherapy",      # 159 samples       61360 features      2 classes
    # "Dorothea",                             # 1150 samples      100000 features     2 classes
]


def main():
    # Cross-validation params
    cv = 10
    n_jobs = 5
    seed = 42
    results_dir = "./results/regression"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    overall_scores = pd.DataFrame()

    bar_position = 0
    progress_bar = tqdm(datasets, position=bar_position)
    for dataset in progress_bar:
        dataset, X, y = dataset
        progress_bar.set_description("Analysis of dataset: %s" % dataset)

        verbose_scores = pd.DataFrame()
        summary_scores = pd.DataFrame()

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        # X, y, n_classes = load_openml_dataset(dataset_name=dataset)
        n_classes = len(set(y))

        # clf = RidgeClassifier(random_state=seed)
        regr = Ridge(random_state=seed)
        est = Ridge(random_state=seed)

        fs_lap_score = SelectKBest(SKF_lap)
        fs_SPEC = SelectKBest(SKF_spec)
        fs_MCFS = SelectKBest(SKF_mcfs)
        fs_NDFS = SelectKBest(SKF_ndfs)
        fs_UDFS = SelectKBest(SKF_udfs)

        sc = StandardScaler()
        pipelines = {
            "NO-FS": Pipeline([("sc", sc), ("est", clone(est))]),
            "DFE": Pipeline([("sc", sc), ("fs", DFE(clone(regr), base_score=0.9)), ("est", clone(est))]),
            "RFE": Pipeline([("sc", sc), ("fs", RFE(clone(regr), verbose=1)), ("est", clone(est))]),
            "lap_score": Pipeline([("sc", sc), ("fs", fs_lap_score), ("est", clone(est))]),
            "SPEC": Pipeline([("sc", sc), ("fs", fs_SPEC), ("est", clone(est))]),
            "NDFS": Pipeline([("sc", sc), ("fs", fs_NDFS), ("est", clone(est))]),
            "UDFS": Pipeline([("sc", sc), ("fs", fs_UDFS), ("est", clone(est))]),
            # "MCFS": Pipeline([("sc", sc), ("fs", fs_MCFS), ("est", clone(est))]),
            # "NO-FS": Pipeline([("sc", sc), ("est", clone(est))]),
        }

        k_best_list = ["lap_score",
                       "SPEC",
                       "NDFS",
                       "UDFS",
                       "MCFS", ]

        for method, pipeline in pipelines.items():

            # for feature selection methods that need to specify the number
            # of features to select a priori, just pick the number of features
            # chosen by DFE
            if method in k_best_list:
                setattr(pipeline.steps[1][1], "k", int(round(k_select)))
            if method == 'RFE':
                setattr(pipeline.steps[1][1], "n_features_to_select", int(round(k_select)))

            # cross validation
            scores = cross_validate(pipeline, X, y, n_jobs=n_jobs, cv=cv,
                                    return_train_score=True, return_estimator=True)

            # save results
            try:
                n_features = [estimator.steps[-1][1].coef_.shape[0] for estimator in scores["estimator"]]
            except:
                n_features = [estimator.steps[-1][1].feature_importances_.shape[0] for estimator in scores["estimator"]]

            if method != 'NO-FS':
                try:
                    features = [estimator.steps[1][1].ranking_ for estimator in scores["estimator"]]
                except:
                    features = [estimator.steps[1][1].scores_ for estimator in scores["estimator"]]
            else:
                features = [np.ones(n_features[0]).astype(int) for _ in range(len(n_features))]

            scores["n_features_selected"] = n_features
            scores["features_selected"] = features
            scores["estimator"] = method

            scores = pd.DataFrame.from_records(scores)
            verbose_scores = pd.concat([verbose_scores, scores], ignore_index=True)
            verbose_scores.to_csv(os.path.join(results_dir, dataset + "_verbose.csv"))

            summary = {
                "dataset": dataset,
                "samples": X.shape[0],
                "features": X.shape[1],
                "classes": n_classes,
                "method": method,
                "avg fit time": np.average(scores["fit_time"]),
                "sem fit time": scipy.stats.sem(scores["fit_time"]),
                "avg train score": np.average(scores["train_score"]),
                "sem train score": scipy.stats.sem(scores["train_score"]),
                "avg test score": np.average(scores["test_score"]),
                "sem test score": scipy.stats.sem(scores["test_score"]),
                "avg n features": np.average(n_features),
                "sem n features": scipy.stats.sem(n_features)
            }

            summary = pd.Series(summary)
            summary_scores = pd.concat([summary_scores, summary], axis=1, ignore_index=True)
            summary_scores.to_csv(os.path.join(results_dir, dataset + "_summary.csv"))

            overall_scores = pd.concat([overall_scores, summary], axis=1, ignore_index=True)
            overall_scores.T.to_csv(os.path.join(results_dir, "overall_results.csv"))

            if method == "DFE":
                k_select = np.average(n_features)


if __name__ == "__main__":
    sys.exit(main())
