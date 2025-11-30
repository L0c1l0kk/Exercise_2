import pandas as pd
import numpy as np

class regtree:
    
    
    dims=tuple()
    boundaries = np.array([])
    features = np.array([])
    averages = np.array([])
    samples = np.array([])
    
    def fit(self, X,y, max_depth:int=3, min_size:int=10):
        
        #Initialization
        X=np.asarray(X)
        y=np.asarray(y)

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                f"X contains non-numeric data (dtype: {X.dtype}). "
                "All features must be numeric. Please encode categorical features before fitting."
            )
        
        self.dims = tuple(X.shape)
        self.features = np.zeros(2**(max_depth+1) - 1, dtype=int)
        self.boundaries = np.zeros(2**(max_depth+1) - 1)
        self.averages = np.zeros(2**(max_depth+1) - 1)
        self.samples = np.zeros(2**(max_depth+1) - 1, dtype=int)
        self.samples[0] = self.dims[0]
        
        elements = np.empty(2**(max_depth+1) - 1,dtype=object)
        elements[0] = np.array([i for i in range(self.dims[0])])
        
        #Loop over levels
        depth=0
        while(depth<max_depth):
            n_leaves=2**depth
            node_idx_start=n_leaves-1
            node_idx_end=2*node_idx_start
            
            no_splits=True
            
            #Loop over (current) leaf nodes
            for node_idx in range(node_idx_start, node_idx_end + 1):
                
                # Indices of elements in current node
                elem_indices = elements[node_idx]
                if elem_indices is None or len(elem_indices) == 0:
                    continue
                
                best_split_loss=np.inf
                best_split=None
                
                #Loop over features
                for col_idx in range(self.dims[1]):
                    
                    #Loop over possible splits and find the best one
                    unique_values=np.unique(X[elem_indices,col_idx])
                    for boundary in unique_values:
                        left_mask = X[elem_indices,col_idx]<=boundary
                        right_mask = X[elem_indices,col_idx]>boundary
                        left_indices = elem_indices[left_mask]
                        right_indices = elem_indices[right_mask]
                        
                        left_size=len(left_indices)
                        right_size=len(right_indices)
                        parent_size=len(elem_indices)
                        
                        if(left_size<min_size or right_size<min_size):
                            continue
                        
                        #Mean squared error loss
                        mse_left=np.var(y[left_indices])
                        mse_right=np.var(y[right_indices])
                        
                        #Weighted MSE loss (by size of child node/size of parent node)
                        loss=left_size/parent_size*mse_left + right_size/parent_size*mse_right
                        if(loss<best_split_loss):
                            best_split_loss=loss
                            best_split=(col_idx,boundary,left_indices,right_indices)
                            
                #Perform split if found
                if(best_split is not None):
                    feature, boundary, left_indices, right_indices = best_split
                    
                    self.features[node_idx] = feature
                    self.boundaries[node_idx] = boundary
                    self.samples[2*node_idx+1] = len(left_indices)
                    self.samples[2*node_idx+2] = len(right_indices)
                    elements[2*node_idx+1] = left_indices
                    elements[2*node_idx+2] = right_indices
                    no_splits=False
                else:
                    self.features[node_idx] = -1
                    self.averages[node_idx] = np.mean(y[elem_indices])
            if(no_splits):
                break
            depth+=1
            # Set node averages for leaf nodes at max depth
            if(depth==max_depth):
                for node_idx in range(node_idx_start, node_idx_end + 1):
                    elem_indices = elements[node_idx]
                    if elem_indices is not None and len(elem_indices) > 0:
                        self.features[node_idx] = -1
                        self.averages[node_idx] = np.mean(y[elem_indices])
    
    
    
    
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
