import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from decision_tree_regressor import DecisionTreeRegressor
import warnings

class RandomForestRegressor:
    def __init__(self, n_trees:int=100, max_depth:int=5, min_size:int=1, n_jobs:int=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        #Input validation
        if np.isnan(X).any():
            warnings.warn("NaN values detected and removed.", RuntimeWarning)
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
        
        X=X.astype(np.float32)
        y=y.astype(np.float32)
        
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                f"X contains non-numeric data (dtype: {X.dtype}). "
                "All features must be numeric. Please encode categorical features before fitting."
            )
            
        
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(X, y,i) for i in range(self.n_trees)
        )
        return self

    def _fit_single_tree(self, X, y,seed):
        tree = DecisionTreeRegressor()
        
        np.random.seed(seed)
        bootstrap_indices = np.random.choice(len(X), size= len(X), replace=True)
        tree.fit(X[bootstrap_indices], y[bootstrap_indices], max_depth=self.max_depth, min_size=self.min_size, random_features=True)
        return tree

    def predict(self, X):
        predictions = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        ))
        return np.mean(predictions, axis=0)