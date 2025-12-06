import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from regtree import regtree

class RandomForestRegressor:
    def __init__(self, n_trees:int=100, max_depth:int=5, min_size:int=3, n_jobs:int=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        bootstrap_indices = np.random.choice(len(X), size=(self.n_trees, len(X)), replace=True)
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(X[bootstrap_indices[i]], y[bootstrap_indices[i]]) for i in range(self.n_trees)
        )

    def _fit_single_tree(self, X, y):
        tree = regtree()
        tree.fit(X, y, max_depth=self.max_depth, min_size=self.min_size, random_features=True)
        return tree

    def predict(self, X):
        predictions = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        ))
        return np.mean(predictions, axis=0)