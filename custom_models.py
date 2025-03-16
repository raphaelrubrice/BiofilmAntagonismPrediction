import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from cuml.neighbors import KNeighborsRegressor

class GPUKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.models_ = {}         # to store a tuple (trained_knn, predictor_indices) for each column
        self.impute_columns_ = [] # indices of columns that need imputation
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Convert input to numpy array; if DataFrame, store column names.
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            X_array = X.values
        else:
            X_array = np.array(X)
        
        n_samples, n_features = X_array.shape
        self.models_ = {}
        self.impute_columns_ = []
        
        # For each feature, check if there are missing values
        for j in range(n_features):
            col = X_array[:, j]
            mask = np.isnan(col)
            if np.any(mask):
                self.impute_columns_.append(j)
                # Use all other columns as predictors
                other_cols = [k for k in range(n_features) if k != j]
                
                # Use only rows where the target column is not missing
                complete_rows = ~mask
                X_train = X_array[complete_rows][:, other_cols]
                y_train = col[complete_rows]
                
                # If there are too few samples to train, store None so that we can later impute by mean.
                if X_train.shape[0] < self.n_neighbors:
                    self.models_[j] = None
                else:
                    # Train cuML's KNeighborsRegressor on the complete cases.
                    knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)
                    knn.fit(X_train, y_train)
                    self.models_[j] = (knn, other_cols)
        return self

    def transform(self, X):
        # Convert input to numpy array (and keep track if original was DataFrame)
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X_array = X.values.copy()
        else:
            X_array = np.array(X, copy=True)
        
        n_samples, n_features = X_array.shape
        
        # For each column that needs imputation, predict missing values.
        for j in self.impute_columns_:
            col = X_array[:, j]
            mask = np.isnan(col)
            if np.any(mask):
                if self.models_.get(j) is not None:
                    knn, other_cols = self.models_[j]
                    # Prepare predictors: use all columns except j.
                    X_missing = X_array[mask][:, other_cols]
                    y_pred = knn.predict(X_missing)
                    # In case the prediction is returned as a GPU array, convert to numpy.
                    if hasattr(y_pred, "to_numpy"):
                        y_pred = y_pred.to_numpy()
                    elif hasattr(y_pred, "get"):
                        y_pred = y_pred.get()
                    X_array[mask, j] = y_pred
                else:
                    # Fallback: if we didn't train a model (due to insufficient data), use column mean.
                    col_mean = np.nanmean(X_array[:, j])
                    X_array[mask, j] = col_mean
                    
        if is_df:
            return pd.DataFrame(X_array, columns=self.feature_names_)
        else:
            return X_array
