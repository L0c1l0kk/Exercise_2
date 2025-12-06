import pandas as pd
import numpy as np
import warnings

class regtree:
    
    
    dims=tuple()
    boundaries = np.array([])
    features = np.array([])
    averages = np.array([])
    samples = np.array([])
    
    def fit(self, X, y, max_depth: int = 5, min_size: int = 10, random_features: bool = False):
        
        # Initialization
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if np.isnan(X).any():
            warnings.warn("NaN values detected and removed.", RuntimeWarning)
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
        
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                f"X contains non-numeric data (dtype: {X.dtype}). "
                "All features must be numeric. Please encode categorical features before fitting."
            )
        
        n_samples, n_features = X.shape
        self.dims = (n_samples, n_features)
        
        max_nodes = 2**(max_depth + 1) - 1
        self.features = np.full(max_nodes, -1, dtype=np.int16)
        self.boundaries = np.zeros(max_nodes, dtype=np.float32)
        self.averages = np.zeros(max_nodes, dtype=np.float32)
        self.samples = np.zeros(max_nodes, dtype=np.uint16)
        self.samples[0] = n_samples
        
        node_masks = np.zeros((max_nodes, n_samples), dtype=bool)
        node_masks[0, :] = True
        
        if random_features:
            n_random_features = max(1, int(np.sqrt(n_features)))
        
        # Loop over levels
        depth = 0
        while depth < max_depth:
            n_leaves = 2**depth
            node_idx_start = n_leaves - 1
            node_idx_end = 2 * node_idx_start
            
            no_splits = True
            
            # Loop over (current) leaf nodes
            for node_idx in range(node_idx_start, node_idx_end + 1):
                mask = node_masks[node_idx]
                
                if(not mask.any()):
                    continue
                
                best_split_loss = np.inf
                best_split = None
                
                # Select features to consider for split (for random forest)
                if random_features:
                    feature_indices = np.random.choice(n_features, size=n_random_features, replace=False)
                else:
                    feature_indices = np.arange(n_features)
                
                X_node = X[mask]
                y_node = y[mask]
                node_size = mask.sum()
                
                # Loop over features
                for col_idx in feature_indices:
                    feature_vals = X_node[:, col_idx]
                    
                    unique_values = np.unique(feature_vals)
                    
                    if len(unique_values) <= 1:
                        continue
                    
                    # Find possible splits
                    splits = (unique_values[:-1] + unique_values[1:]) / 2
                    left_masks = feature_vals[:, None] <= splits[None, :]
                    left_sizes = left_masks.sum(axis=0)
                    right_sizes = node_size - left_sizes
                    
                    valid_splits = (left_sizes >= min_size) & (right_sizes >= min_size)
                    
                    if not valid_splits.any():
                        continue
                    
                    valid_indices = np.where(valid_splits)[0]
                    
                    #Loop over possible (valid) splits
                    for idx in valid_indices:
                        left_mask_local = left_masks[:, idx]
                        
                        y_left = y_node[left_mask_local]
                        y_right = y_node[~left_mask_local]
                        
                        left_size = len(y_left)
                        right_size = len(y_right)
                        parent_size = node_size
                        
                        # Mean squared error loss
                        mse_left = np.var(y_left)
                        mse_right = np.var(y_right)
                        
                        # Weighted MSE loss (by size of child node/size of parent node)
                        loss = left_size / parent_size * mse_left + right_size / parent_size * mse_right
                        
                        if loss < best_split_loss:
                            best_split_loss = loss
                            best_split = (col_idx, splits[idx], left_mask_local)
                
                # Perform split if found
                if best_split is not None:
                    feature, boundary, left_mask_local = best_split
                    
                    self.features[node_idx] = feature
                    self.boundaries[node_idx] = boundary
                    
                    global_left_mask = mask.copy()
                    global_left_mask[mask] = left_mask_local
                    
                    global_right_mask = mask.copy()
                    global_right_mask[mask] = ~left_mask_local
                    
                    node_masks[2 * node_idx + 1] = global_left_mask
                    node_masks[2 * node_idx + 2] = global_right_mask
                    self.samples[2 * node_idx + 1] = global_left_mask.sum()
                    self.samples[2 * node_idx + 2] = global_right_mask.sum()
                    
                    no_splits = False
                else:
                    self.features[node_idx] = -1
                    self.averages[node_idx] = np.mean(y_node)
                    node_masks[2 * node_idx + 1] = False
                    node_masks[2 * node_idx + 2] = False
            
            depth += 1
            
            # Set node averages for leaf nodes at max depth
            if no_splits or depth == max_depth:
                for node_idx in range(node_idx_start, node_idx_end + 1):
                    if node_masks[node_idx].any():
                        self.features[node_idx] = -1
                        self.averages[node_idx] = np.mean(y[node_masks[node_idx]])
            if no_splits:
                break
    
    
    
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
    
        predictions = np.zeros(len(X))
    
        for i, sample in enumerate(X):
            node_idx = 0
        
            while self.features[node_idx] != -1: 
                feature = self.features[node_idx]
                boundary = self.boundaries[node_idx]
            
                if 2*node_idx+1 >= len(self.features):
                    break
                elif sample[feature] <= boundary:
                    node_idx = 2 * node_idx + 1 
                else:
                    node_idx = 2 * node_idx + 2 
        
            predictions[i] = self.averages[node_idx]
    
        return predictions
    
    def data(self):
        return {
            "dims": self.dims,
            "features": self.features,
            "boundaries": self.boundaries,
            "averages": self.averages
        }
    
    def print_rules(self,feature_names=None):
        
        if(feature_names is None):
            feature_names=["feature_"+str(i) for i in range(self.dims[1])]
        
        stack=[(0,0,True)]
        while(len(stack)>0):
            node_idx, depth, left=stack.pop()
            print("|   "*depth, end="")
            print("|--- ",end="")
            if self.features[node_idx]==-1:
                print("class: ",self.averages[node_idx], f"(samples: {self.samples[node_idx]})",end="")
            else:
                if left:
                    print(f"{feature_names[self.features[node_idx]]} <= {self.boundaries[node_idx]}",end="")
                    stack.append((2*node_idx+2, depth+1, False))
                    stack.append((2*node_idx+2, depth+1, True))
                    stack.append((node_idx, depth, False))
                    stack.append((2*node_idx+1, depth+1, False))
                    stack.append((2*node_idx+1, depth+1, True))
                else:
                    print(f"{feature_names[self.features[node_idx]]} > {self.boundaries[node_idx]}",end="")

            print("\n", end="")
