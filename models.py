import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone, BaseEstimator
from sklearn.model_selection import train_test_split
from joblib import Memory, Parallel, delayed

class NaNFilter(TransformerMixin):
    def __init__(self, transformer: TransformerMixin):
        super().__init__()
        self.transformer = transformer
        self.isnan_mask_ = None
        self.is_pandas_input_ = False

    def fit(self, X, y=None, **fit_params):
        # Detect pandas DataFrame
        self.is_pandas_input_ = isinstance(X, pd.DataFrame)
        # Compute row-wise NaN mask
        mat = X.values if self.is_pandas_input_ else X
        self.isnan_mask_ = np.isnan(mat).any(axis=1)
        # Filter out NaNs
        X_clean = X[~self.isnan_mask_]
        if y is not None:
            y_clean = y[~self.isnan_mask_]
            self.transformer.fit(X_clean, y_clean, **fit_params)
        else:
            self.transformer.fit(X_clean, **fit_params)
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        mat = X.values if is_df else X
        mask = np.isnan(mat).any(axis=1)
        X_clean = X[~mask]
        X_trans = self.transformer.transform(X_clean)
        # Reconstruct result
        if is_df:
            # Handle DataFrame output
            if isinstance(X_trans, pd.DataFrame):
                X_out = X.copy()
                X_out.loc[~mask, :] = X_trans.values
            else:
                X_out = X.copy()
                X_out.loc[~mask, :] = X_trans
        else:
            X_out = mat.copy()
            X_out[~mask] = X_trans
        # Wrap back to DataFrame if input was DataFrame
        if is_df:
            return pd.DataFrame(
                X_out,
                index=X.index,
                columns=X.columns
            )
        return X_out

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y, **fit_params)
        return self.transform(X)
    
class StratifiedRegressor(BaseEstimator):
    """
    Implements stratified regression using the provided estimator.
    """
    def __init__(self, instantiated_estimator, 
                 mode: str = 'quantile', 
                 ranges: list = [0.2, 0.4, 0.6, 0.8],
                 n_jobs: int = 1,
                 parallel: bool = True,
                 random_state: int = 6262):
        super().__init__()
        self.base_estimator = instantiated_estimator
        self.mode = mode
        if self.mode != 'quantile':
            assert ranges is not None, f"You must specify ranges when not using quantile mode"
        self.ranges = ranges
        self.n_estimators = len(self.ranges) + 2 # (n_values + 1) + oracle
        self.random_state = random_state
        self.n_jobs = n_jobs # Per estimator threads
        self.parallel = parallel # Fit all estimators simultaneously
    
    def split(self, X, y):
        subset_sizes = 1 / self.n_estimators
        X_oracle, X_strat, y_oracle, y_strat = train_test_split(X, y, 
                                                                train_size=subset_sizes, 
                                                                shuffle=True, 
                                                                random_state=self.random_state)
        splitted = {'oracle': (X_oracle,y_oracle)}
        if self.mode == 'quantile':
            quantiles = [0.2, 0.4, 0.6, 0.8] # 5 equally large quantiles
            self.ranges = [np.quantile(y_strat, q) for q in quantiles]
            print(f"\nQuantile ranges are: {self.ranges}")
            
        splitted['class1'] = (X_strat[y_strat < self.ranges[0]],y_strat[y_strat < self.ranges[0]])
        for i, val in enumerate(self.ranges):
            mask = (y_strat >= self.ranges[i - 1]) & (y_strat < val)
            splitted[f'class{i}'] = (X_strat[mask],y_strat[mask])
        splitted[f'class{len(self.ranges)}'] = (X_strat[y_strat >= self.ranges[-1]],y_strat[y_strat >= self.ranges[-1]])
        return splitted
    
    def fit(self, X, y, fit_params={}):
        splitted = self.split(X, y)
    
        if self.parallel:
            result_list = Parallel(n_jobs=self.n_estimators)(
                    delayed(fit_submodel)(self.base_estimator, val[0], val[1])
                    for key, val in splitted.items()
                )
            self.estimators = {key:result_list[i] for i, key in enumerate(splitted.keys())}
        else:
            self.estimators = {}
            for key, val in splitted.items():
                X_train, Y_train = val[0], val[1]
                self.estimators[key] = fit_submodel(self.base_estimator, X_train, Y_train)
        self.booster_ = BoosterWrapper(self.estimators)
        return self

    def get_stratification_masks(self, X, return_y_oracle=False):
        y_oracle = self.estimators['oracle'].predict(X)
        strat_keys = [key for key in self.estimators.keys() if key != 'oracle']

        strat_masks = {}
        for i, key in enumerate(strat_keys):
            upper_bound = self.ranges[i]
            if i != 0 and i != len(self.ranges) - 1:
                mask = (y_oracle >= self.ranges[i -1]) & (y_oracle < upper_bound)
            elif i == 0:
                mask = y_oracle < upper_bound
            elif i == len(self.ranges) - 1:
                mask = y_oracle >= upper_bound
            strat_masks[key] = mask
        if not return_y_oracle:
            y_oracle = None
        return strat_masks, y_oracle

    def predict(self, X, 
                return_y_oracle: bool = False,
                return_used_estimators: bool = False):
        strat_masks, y_oracle = self.get_stratification_masks(X, 
                                                              return_y_oracle=return_y_oracle)
        y_pred = np.zeros((X.shape[0],1))
        estimator_track = np.zeros((X.shape[0],1))
        if self.parallel: # Parallel inference
            result_list = Parallel(n_jobs=self.n_estimators-1)(
                delayed(predict_submodel)(self.estimators[key], X[mask])
                for key, mask in strat_masks.items()
                )
            for i, item in strat_masks.items():
                key, mask = item[0], item[1]
                y_pred[mask] = result_list[i]
                estimator_track[mask] = [key] * mask.shape[0]
        else:
            for key, mask in strat_masks.items():
                y_pred[mask] = self.estimators[key].predict(X[mask])
                estimator_track[mask] = [key] * mask.shape[0]
        if y_oracle is not None:
            if return_used_estimators:
                return y_pred, y_oracle, return_used_estimators
            return y_pred, y_oracle
        if return_used_estimators:
            return y_pred, return_used_estimators
        return y_pred

def fit_submodel(base_estimator, X, Y):
    return clone(base_estimator).fit(X, Y)

def predict_submodel(estimator, X):
    return estimator.predict(X)

def BoosterWrapper():
    """
    Small wrapper to keep the code as is in other parts of the project 
    when using the startified regressor.
    """
    def __init__(self, estimator_dict: dict):
        self.estimators = estimator_dict
    
    def save_model(self, path):
        splitted_path = path.split('.')
        extension = splitted_path[-1]
        before_extension = path[:path.index('.' + extension)]
        for model in self.estimators.keys():
            new_path = f"{before_extension}_{model}." + extension
            model.booster_.save_model(new_path)