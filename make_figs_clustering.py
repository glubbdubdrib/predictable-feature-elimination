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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


random_state = 42
n_jobs = 2
cv = 2
results_dir = "./results/clustering/"
file_list = glob.glob(results_dir + '**verbose.csv')

test_score = []
train_score = []
feature_size = []
feature_selector = []
estimator = []
fit_time = []
dataset_name = []

for fi, file in enumerate(file_list):
    if 'gas-drift' not in file:
        continue

    dataset = file.split('/')[3].split('_verbose.csv')[0]
    X, y, n_classes = load_openml_dataset(dataset_name=dataset)
    # X, y = datasets[dataset]
    results = pd.read_csv(file, index_col=0)
    features_selected = results['features_selected']

    n_selected = []
    for r in range(results.shape[0]):
        print(f'{fi+1}/{len(file_list)} - {r+1}/{results.shape[0]}')

        selector = results['estimator'][r]
        time = results['fit_time'][r]
        feature_set_str = features_selected[r].replace('[', '').replace(']', '').split()
        feature_set = np.array([int(i) for i in feature_set_str])

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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
            continue
        elif selector == 'NO-FS':
            X_train = X_train
            X_test = X_test
        else:
            X_train = X_train[:, feature_set[:n_selected[r % 10]]]
            X_test = X_test[:, feature_set[:n_selected[r % 10]]]

        n_clusters = len(set(y)) * 3
        clustering_algs = [
            # KMeans(n_clusters=n_clusters, random_state=random_state),
            DBSCAN(),
            # AgglomerativeClustering(n_clusters=n_clusters),
        ]

        if selector == 'DFE':
            selector = 'PFE'
        for clustering_alg in clustering_algs:
            train_pred = clustering_alg.fit_predict(X_train)
            # train_pred = clustering_alg.predict(X_train)
            # test_pred = clustering_alg.predict(X_test)
            train_score_r = silhouette_score(X_train, train_pred)
            # test_score_r = silhouette_score(X_test, test_pred)

            # test_score.append(test_score_r)
            train_score.append(train_score_r)
            feature_size.append(X_train.shape[1])
            feature_selector.append(selector)
            estimator.append(clustering_alg.__class__.__name__)
            fit_time.append(time)
            dataset_name.append(dataset)

results = pd.DataFrame({
    'dataset': dataset_name,
    'silhouette score': train_score,
    # 'train_score': train_score,
    'feature_size': feature_size,
    'algorithm': feature_selector,
    'estimator': estimator,
    'training time (s)': fit_time,
})


import matplotlib
sns.set_style('whitegrid')

for dataset in set(results['dataset']):
    # miny = results[results['dataset'] == dataset]['silhouette score'].min() - 0.005
    # maxy = results[results['dataset'] == dataset]['silhouette score'].max() + 0.005
    for estimator in set(results['estimator']):
        res = results[(results['dataset'] == dataset) & (results['estimator'] == estimator)]

        perc_test_f1 = []
        for algorithm in sorted(set(res['algorithm'])):
            perc_test_f1.extend(res[res['algorithm'] == algorithm]['silhouette score'].values - res[res['algorithm'] == 'NO-FS']['silhouette score'].values)

        res['% silhouette score'] = perc_test_f1

        resPFE = res[res['algorithm'] == 'PFE']
        resRFE = res[res['algorithm'] == 'RFE']
        resALL = res[(res['algorithm'] != 'PFE') & (res['algorithm'] != 'RFE') & (res['algorithm'] != 'NO-FS')]

        res = pd.concat([resPFE, resALL, resRFE])

        plt.figure(figsize=[4, 3])
        sns.scatterplot(x='training time (s)', y='% silhouette score', hue='algorithm', data=res, alpha=0.4)
        sns.scatterplot(x='training time (s)', y='% silhouette score', hue='algorithm', data=res[res['algorithm'] == 'PFE'], alpha=1)
        plt.legend().set_visible(False)
        plt.title(f'{dataset} - {estimator}')
        # plt.ylim([miny, maxy])
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{results_dir}{dataset}_{estimator}_clustering.png')
        plt.savefig(f'{results_dir}{dataset}_{estimator}_clustering.pdf')
        plt.show()

plt.figure(figsize=[4, 5])
sns.scatterplot(x='training time (s)', y='silhouette score', hue='algorithm', data=res, alpha=0.4)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=3)
plt.tight_layout()
plt.savefig(f'{results_dir}legend_clustering.png', bbox_inches=matplotlib.transforms.Bbox([[0.4, 2.4], [4, 3.3]]))
plt.savefig(f'{results_dir}legend_clustering.pdf', bbox_inches=matplotlib.transforms.Bbox([[0.4, 2.4], [4, 3.3]]))
plt.show()
