import copy
import glob
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mp
from lazygrid.datasets import load_openml_dataset
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

random_state = 42
n_jobs = 2
cv = 2
results_dir = "./results/regression/"
file_list = glob.glob(results_dir + '**verbose.csv')

test_score = []
train_score = []
feature_size = []
feature_selector = []
estimator = []
fit_time = []
dataset_name = []

for fi, file in enumerate(file_list):
    if 'Mercedes_Benz_Greener_Manufacturing' not in file:
        continue

    dataset = file.split('/')[3].split('_verbose.csv')[0]
    X, y, n_classes = load_openml_dataset(dataset_name=dataset)
    # X, y = datasets[dataset]
    results = pd.read_csv(file, index_col=0)
    features_selected = results['features_selected']

    n_selected = []
    for r in range(results.shape[0]):
        print(f'{fi}/{len(file_list)} - {r}/{results.shape[0]}')
        selector = results['estimator'][r]
        time = results['fit_time'][r]
        feature_set_str = features_selected[r].replace('[', '').replace(']', '').split()
        feature_set = np.array([int(i) for i in feature_set_str])

        skf = KFold(n_splits=10, shuffle=True, random_state=42)
        train_index, test_index = [split for split in skf.split(X, y)][r % 10]

        sc = StandardScaler()
        sc.fit(X[train_index])
        X_train = sc.transform(X[train_index])
        X_test = sc.transform(X[test_index])

        if selector == 'DFE':
            X_train = X_train[:, feature_set == 1]
            X_test = X_test[:, feature_set == 1]
            n_selected.append(X_train.shape[1])
        elif selector == 'RFE':
            X_train = X_train[:, feature_set == 1]
            X_test = X_test[:, feature_set == 1]
            n_selected.append(X_train.shape[1])
            # rfe = RFE(RidgeClassifier(random_state=random_state), n_features_to_select=n_selected[r % 10])
            # rfe.fit(X_train, y[train_index])
            # X_train = X_train[:, rfe.ranking_ == 1]
            # X_test = X_test[:, rfe.ranking_ == 1]
        elif selector == 'NO-FS':
            X_train = X_train
            X_test = X_test
        else:
            X_train = X_train[:, feature_set[:n_selected[r % 10]]]
            X_test = X_test[:, feature_set[:n_selected[r % 10]]]

        classifiers = [
            # RandomForestClassifier(random_state=random_state),
            Ridge(random_state=random_state),
            DecisionTreeRegressor(random_state=random_state),
            # SVC(random_state=random_state),
        ]

        if selector == 'DFE':
            selector = 'PFE'
        for classifier in classifiers:
            classifier.fit(X_train, y[train_index])
            train_pred = classifier.predict(X_train)
            test_pred = classifier.predict(X_test)
            train_score_r = r2_score(y[train_index], train_pred)
            test_score_r = r2_score(y[test_index], test_pred)

            test_score.append(test_score_r)
            train_score.append(train_score_r)
            feature_size.append(X_train.shape[1])
            feature_selector.append(selector)
            estimator.append(classifier.__class__.__name__)
            fit_time.append(time)
            dataset_name.append(dataset)

results = pd.DataFrame({
    'dataset': dataset_name,
    'test R2': test_score,
    'train_score': train_score,
    'feature_size': feature_size,
    'algorithm': feature_selector,
    'estimator': estimator,
    'training time (s)': fit_time,
})


import matplotlib
sns.set_style('whitegrid')

for dataset in set(results['dataset']):
    # miny = results[results['dataset'] == dataset]['test F1'].min() - 0.005
    # maxy = results[results['dataset'] == dataset]['test F1'].max() + 0.005
    for estimator in set(results['estimator']):
        res = results[(results['dataset'] == dataset) & (results['estimator'] == estimator)]

        perc_test_f1 = []
        for algorithm in sorted(set(res['algorithm'])):
            perc_test_f1.extend(res[res['algorithm'] == algorithm]['test R2'].values - res[res['algorithm'] == 'NO-FS']['test R2'].values)

        res['% test R2'] = perc_test_f1

        resPFE = res[res['algorithm']=='PFE']
        resRFE = res[res['algorithm']=='RFE']
        resALL = res[(res['algorithm']!='PFE') & (res['algorithm']!='RFE') & (res['algorithm']!='NO-FS')]

        res = pd.concat([resPFE, resALL, resRFE])

        ds = dataset.split('_')[0]

        plt.figure(figsize=[4, 3])
        sns.scatterplot(x='training time (s)', y='% test R2', hue='algorithm', data=res, alpha=0.4)
        sns.scatterplot(x='training time (s)', y='% test R2', hue='algorithm', data=res[res['algorithm']=='PFE'], alpha=1)
        plt.legend().set_visible(False)
        plt.title(f'{ds} - {estimator}')
        # plt.ylim([miny, maxy])
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{results_dir}{ds}_{estimator}_regression.png')
        plt.savefig(f'{results_dir}{ds}_{estimator}_regression.pdf')
        plt.show()

plt.figure(figsize=[4, 5])
sns.scatterplot(x='training time (s)', y='test R2', hue='algorithm', data=res, alpha=0.4)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=3)
plt.tight_layout()
plt.savefig(f'{results_dir}legend_regression.png', bbox_inches=matplotlib.transforms.Bbox([[0.6, 2.4], [4, 3.3]]))
plt.savefig(f'{results_dir}legend_regression.pdf', bbox_inches=matplotlib.transforms.Bbox([[0.6, 2.4], [4, 3.3]]))
plt.show()


# results = pd.read_csv(results_file, index_col=0)
# results["avg test error"] = 1 - results["avg test score"]
# results["sem test error"] = results["sem test score"]
# results["size"] = results["samples"] * results["features"]
# dataset_set = results[["dataset", "samples", "features", "size"]].drop_duplicates().sort_values("size")
# n_rows = int(np.sqrt(len(dataset_set)))
# dataset_set.drop([0, 48, 64], axis=0, inplace=True)
#
# res = results.copy()
# ok_list = [
#     "yeast_ml8",
#     "scene",
#     "isolet",
#     "gina_agnostic",
#     "gas-drift",
#     "letter",
# ]
# idx = []
# for i in range(len(res)):
#     if res["dataset"][i] in ok_list:
#         idx.append(i)
# res = res.iloc[idx]
# res["size"] = res["samples"] * res["features"]
#
# plt.figure()
# sns.lineplot(x="size", y="avg fit time", data=res, hue="method")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
# plt.show()

# def is_pareto_efficient_simple(costs):
#     """
#     Find the pareto-efficient points
#     :param costs: An (n_points, n_costs) array
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype = bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
#             is_efficient[i] = True  # And keep self
#     return is_efficient
#
#
# plt.figure(figsize=[8, 12])
# for i, (k, dataset) in enumerate(dataset_set.iterrows()):
#     d = results[results["dataset"] == dataset["dataset"]]
#     is_pareto = is_pareto_efficient_simple(d[d["method"] != "NO-FS"][["avg fit time", "avg test error"]].values)
#     current_palette = sns.color_palette(n_colors=d.shape[0])
#     plt.subplot(3, 2, i+1)
#     lines = []
#     labels = []
#     for xi, yi, exi, eyi, method, c in zip(d["avg fit time"], d["avg test error"], d["sem fit time"], d["sem test error"], d["method"], current_palette):
#         l = plt.errorbar(xi, yi, xerr=exi, yerr=eyi, fmt='o', ecolor=c, c=c, alpha=0.5)
#         e = mp.Ellipse((xi, yi), width=exi*2, height=eyi*2, color=c, alpha=0.5)
#         plt.gca().add_patch(e)
#         lines.append(l)
#         labels.append(method)
#     pareto_x = d[d["method"] != "NO-FS"]["avg fit time"][is_pareto].values
#     pareto_y = d[d["method"] != "NO-FS"]["avg test error"][is_pareto].values
#     x_sorted = np.argsort(pareto_x)
#     pareto_x = pareto_x[x_sorted]
#     pareto_y = pareto_y[x_sorted]
#     plt.plot(pareto_x, pareto_y, c='k', ls='--', alpha=0.8)
#     plt.scatter(pareto_x, pareto_y, c='k', marker='*', s=80)
#     plt.title(f'{dataset["dataset"]} [S={dataset["samples"]}, F={dataset["features"]}]')
#     plt.xscale("log")
#     if i >= 4:
#         plt.xlabel("fit time")
#     if i % 2 == 0:
#         plt.ylabel("test error")
# plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
# plt.tight_layout()
# plt.savefig("summary.png")
# plt.savefig("summary.pdf")
# plt.show()
