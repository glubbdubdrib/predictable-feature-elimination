import sys
import scipy
import os

from sklearn.datasets import make_blobs, make_circles
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dfe.dfe import DFE


def main():
    random_state = 42
    X, y = make_blobs(n_samples=500, random_state=random_state)

    # est = Ridge(random_state=random_state)
    est = Ridge()
    dfe = DFE(clone(est), base_score=0.99, n_splits=0, n_features_to_select=20, random_state=1)
    dfe.fit(X.T, y)
    X2 = dfe.transform(X.T).T

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.scatter(X2[:, 0], X2[:, 1], c='r', marker='x', s=200, label='selected samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig('samples.png')
    plt.show()
    return


if __name__ == "__main__":
    sys.exit(main())
