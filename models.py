import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone, BaseEstimator
from sklearn.model_selection import train_test_split
from joblib import Memory, Parallel, delayed

class NaNFilter(TransformerMixin):
    """
    Wraps a TransformerMixin to apply independently per column,
    ignoring NaNs during fit/transform. Stores a fitted transformer
    per column to be reused.
    """

    def __init__(self, transformer: TransformerMixin):
        self.base_transformer = transformer
        self.col_transformers_ = {}  # column name or index -> fitted transformer
        self.__sklearn_tags__ = self.base_transformer.__sklearn_tags__

    def fit(self, X, y=None, **fit_params):
        df, is_df = (X, True) if isinstance(X, pd.DataFrame) else (pd.DataFrame(X), False)
        y_series = pd.Series(y, index=df.index) if y is not None else None

        for col in df.columns:
            col_data = df[col]
            mask = ~pd.isna(col_data)
            if mask.sum() == 0:
                continue  # skip columns with all NaNs

            trans = clone(self.base_transformer)
            X_col = col_data[mask].to_frame()
            y_col = y_series[mask] if y_series is not None else None
            if y_col is not None and hasattr(trans, "fit_transform"):
                trans.fit(X_col, y_col, **fit_params)
            else:
                trans.fit(X_col, **fit_params)
            self.col_transformers_[col] = trans

        return self

    def transform(self, X):
        df_out, is_df = (X.copy(), True) if isinstance(X, pd.DataFrame) else (pd.DataFrame(X).copy(), False)

        for col in df_out.columns:
            col_data = df_out[col]
            mask = ~pd.isna(col_data)

            if col not in self.col_transformers_ or mask.sum() == 0:
                continue

            trans = self.col_transformers_[col]
            X_col = col_data[mask].to_frame()
            out = trans.transform(X_col)
            df_out.loc[mask, col] = np.array(out).ravel()

        return df_out if is_df else df_out.values

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)
    
class StratifiedRegressor(BaseEstimator):
    """
    Implements stratified regression using the provided estimator.
    """
    def __init__(self, base_estimator, 
                 mode: str = 'quantile', 
                 ranges: list = [0.2, 0.4, 0.6, 0.8],
                 n_jobs: int = 1,
                 parallel: bool = True,
                 mixed_training: bool = False,
                 random_state: int = 6262):
        super().__init__()
        self.base_estimator = base_estimator
        self.mode = mode
        if self.mode != 'quantile':
            assert ranges is not None, f"You must specify ranges when not using quantile mode"
        self.ranges = ranges
        self.n_estimators = len(self.ranges) + 2 # (n_values + 1) + oracle
        self.random_state = random_state
        self.n_jobs = n_jobs # Per estimator threads
        self.parallel = parallel # Fit all estimators simultaneously
        self.mixed_training = mixed_training
        self.__sklearn_tags__ = self.base_estimator.__sklearn_tags__

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
        for i, val in enumerate(self.ranges[1:]):
            i += 1
            mask = (y_strat >= self.ranges[i - 1]) & (y_strat < val)
            splitted[f'class{i+1}'] = (X_strat[mask],y_strat[mask])
        splitted[f'class{len(self.ranges)+1}'] = (X_strat[y_strat >= self.ranges[-1]],y_strat[y_strat >= self.ranges[-1]])
        return splitted
    
    def default_fit(self, splitted, fit_params={}):
        if self.parallel:
            result_list = Parallel(n_jobs=self.n_estimators)(
                    delayed(fit_submodel)(self.base_estimator, val[0], val[1], fit_params)
                    for key, val in splitted.items()
                )
            self.estimators = {key:result_list[i] for i, key in enumerate(splitted.keys())}
        else:
            self.estimators = {}
            for key, val in splitted.items():
                X_train, Y_train = val[0], val[1]
                self.estimators[key] = fit_submodel(self.base_estimator, X_train, Y_train, fit_params)
        self.booster_ = BoosterWrapper(self.estimators)
        return self

    def mixed_fit(self, splitted, fit_params={}):
        X_train, Y_train = splitted['oracle']
        self.estimators['oracle'] = fit_submodel(self.base_estimator, X_train, Y_train, fit_params)
        mixed_splitted = {}
        for key, val in splitted.items():
            if key != 'oracle':
                X_train, Y_train = val[0], val[1]
                Y_oracle = self.estimators['oracle'].predict(X_train)

                mixed_X_train = pd.concatenate([X_train, X_train], axis=0)
                mixed_Y_train = np.concatenate([Y_train, Y_oracle], axis=0)

                mixed_splitted[key] = (mixed_X_train, mixed_Y_train)
    
        if self.parallel:
            result_list = Parallel(n_jobs=self.n_estimators-1)(
                    delayed(fit_submodel)(self.base_estimator, val[0], val[1], fit_params)
                    for key, val in mixed_splitted.items()
                )
            for i, key in enumerate(mixed_splitted.keys()):
                self.estimators[key] = result_list[i]
        else:
            self.estimators = {}
            for key, val in mixed_splitted.items():
                X_train, Y_train = val[0], val[1]
                self.estimators[key] = fit_submodel(self.base_estimator, X_train, Y_train, fit_params)
        self.booster_ = BoosterWrapper(self.estimators)
        return self

    def fit(self, X, y, fit_params={}):
        splitted = self.split(X, y)
        if self.mixed_training:
            return self.mixed_fit(splitted, fit_params)
        return self.default_fit(splitted, fit_params)

    def get_stratification_masks(self, X, return_y_oracle=False):
        y_oracle = self.estimators['oracle'].predict(X)
        # print("\nY_oracle", y_oracle)
        strat_keys = [key for key in self.estimators.keys() if key != 'oracle']
        # print('Strat keys', strat_keys)
        strat_masks = {}
        for i, key in enumerate(strat_keys):
            if i < len(self.ranges):
                upper_bound = self.ranges[i]
            if i != 0 and i < len(self.ranges):
                mask = (y_oracle >= self.ranges[i -1]) & (y_oracle < upper_bound)
            elif i == 0:
                mask = y_oracle < upper_bound
            else:
                mask = y_oracle >= self.ranges[-1]
            if np.sum(mask) != 0:
              strat_masks[key] = mask
        if not return_y_oracle:
            y_oracle = None
        # print('Strat masks keys', strat_masks.keys())
        return strat_masks, y_oracle

    def predict(self, X, 
                return_y_oracle: bool = False,
                return_used_estimators: bool = False):
        strat_masks, y_oracle = self.get_stratification_masks(X, 
                                                              return_y_oracle=return_y_oracle)
        n_masks = len(strat_masks)
        y_pred = np.zeros((X.shape[0],1))
        estimator_track = np.zeros((X.shape[0],1), dtype='<U7')
        if self.parallel: # Parallel inference
            result_list = Parallel(n_jobs=n_masks)(
                delayed(predict_submodel)(self.estimators[key], X[mask])
                for key, mask in strat_masks.items()
                )
            for i, item in enumerate(strat_masks.items()):
                key, mask = item[0], item[1].reshape(-1,1)
                y_pred[mask] = result_list[i]
                estimator_track[mask] = [key] * np.sum(mask)
        else:
            for key, mask in strat_masks.items():
                y_pred[mask] = self.estimators[key].predict(X[mask])
                estimator_track[mask] = [key] * np.sum(mask)
        # print('\nY_pred', y_pred)
        # print("Estimator track", estimator_track)
        if y_oracle is not None:
            if return_used_estimators:
                return y_pred, y_oracle, return_used_estimators
            return y_pred, y_oracle
        if return_used_estimators:
            return y_pred, return_used_estimators
        return y_pred

def fit_submodel(base_estimator, X, Y, fit_params={}):
    return clone(base_estimator).fit(X, Y, **fit_params)

def predict_submodel(estimator, X):
    return estimator.predict(X)

class BoosterWrapper:
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
            self.estimators[model].booster_.save_model(new_path)