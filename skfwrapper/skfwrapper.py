# -*- coding: utf-8 -*-
#
# Copyright 2020 Pietro Barbiero, Alberto Tonda and Giovanni Squillero
# Licensed under the EUPL

from skfeature.function.similarity_based import SPEC, lap_score
from skfeature.function.sparse_learning_based import MCFS, NDFS, UDFS
from skfeature.function.statistical_based import low_variance
from skfeature.utility import sparse_learning
from skfeature.utility.construct_W import construct_W


def SKF_lap(X, y):
    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W(X, **kwargs_W)
    # obtain the scores of features
    score = lap_score.lap_score(X, W=W)
    return lap_score.feature_ranking(score)


def SKF_spec(X, y):
    # specify the second ranking function which uses all except the 1st eigenvalue
    kwargs = {'style': 0}
    # obtain the scores of features
    score = SPEC.spec(X, **kwargs)
    return SPEC.feature_ranking(score, **kwargs)


def SKF_mcfs(X, y):
    # construct affinity matrix
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W(X, **kwargs)
    num_fea = X.shape[1]  # specify the number of selected features
    num_cluster = len(
        set(y))  # specify the number of clusters, it is usually set as the number of classes in the ground truth
    # obtain the feature weight matrix
    Weight = MCFS.mcfs(X, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
    return MCFS.feature_ranking(Weight)


def SKF_ndfs(X, y):
    # construct affinity matrix
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W(X, **kwargs)
    num_cluster = len(
        set(y))  # specify the number of clusters, it is usually set as the number of classes in the ground truth
    # obtain the feature weight matrix
    Weight = NDFS.ndfs(X, W=W, n_clusters=num_cluster)
    return sparse_learning.feature_ranking(Weight)


def SKF_udfs(X, y):
    num_cluster = len(
        set(y))  # specify the number of clusters, it is usually set as the number of classes in the ground truth
    Weight = UDFS.udfs(X, gamma=0.1, n_clusters=num_cluster)
    return sparse_learning.feature_ranking(Weight)
