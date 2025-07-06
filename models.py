import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, RegressorMixin, clone, BaseEstimator
from sklearn.model_selection import train_test_split
from joblib import Memory, Parallel, delayed
from lightgbm import LGBMRegressor, LGBMClassifier

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

class StratifiedRegressor(LGBMRegressor):
    """
    Implements stratified regression using the provided estimator.
    """
    def __init__(self, base_estimator, 
                 mode: str = 'quantile', 
                 ranges: list = [0.2, 0.4, 0.6, 0.8],
                 n_jobs: int = 1,
                 parallel: bool = True,
                 mixed_training: bool = False,
                 router_training: bool = False,
                 base_router=LGBMClassifier(),
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
        self.router_training = router_training
        self.base_router = base_router
        self.is_fitted_ = False
        self.__sklearn_tags__ = self.base_estimator.__sklearn_tags__

    def split(self, X, y):
        if self.router_training:
            X, X_router, y, y_router = train_test_split(X, y, 
                                                        train_size=0.8, 
                                                        shuffle=True, 
                                                        random_state=self.random_state)
            if self.mode == 'quantile':
                quantiles = [0.2, 0.4, 0.6, 0.8] # 5 equally large quantiles
                self.ranges = [np.quantile(y, q) for q in quantiles]
                print(f"\nQuantile ranges are: {self.ranges}")

            splitted = {'class1':(X[y < self.ranges[0]],y[y < self.ranges[0]])}
            for i, val in enumerate(self.ranges[1:]):
                i += 1
                mask = (y >= self.ranges[i - 1]) & (y < val)
                splitted[f'class{i+1}'] = (X[mask],y[mask])
            splitted[f'class{len(self.ranges)+1}'] = (X[y >= self.ranges[-1]],y[y >= self.ranges[-1]])
            return splitted, X_router, y_router
        else:
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
        # self.booster_ = BoosterWrapper(self.estimators)
        self.is_fitted_ = True
        return self

    def mixed_fit(self, splitted, fit_params={}):
        X_train, Y_train = splitted['oracle']
        self.estimators = {'oracle':fit_submodel(self.base_estimator, X_train, Y_train, fit_params)}
        mixed_splitted = {}
        for key, val in splitted.items():
            if key != 'oracle':
                X_train, Y_train = val[0], val[1]
                Y_oracle = self.estimators['oracle'].predict(X_train)

                mixed_X_train = pd.concat([X_train, X_train], axis=0)
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
        # self.booster_ = BoosterWrapper(self.estimators)
        self.is_fitted_ = True
        return self

    def routed_fit(self, X_router, y_router, splitted, fit_params={}):
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
        
        # Gather predictions for all range-specific estimators on unseen data
        predicted_scores = [self.estimators[key].predict(X_router).reshape(-1,1) for key in splitted.keys()]
        self.router_mapping = {i:key for i, key in enumerate(splitted.keys())}
        predicted_scores = np.concatenate(predicted_scores, axis=1)

        # Identify the best estimators based on absolute error
        abs_err = np.abs(predicted_scores - y_router.reshape(-1,1))
        best_estimators = np.argmin(abs_err, axis=1).reshape(-1,1)

        # Train the router to learn to predict the best sub model to use
        self.estimators['router'] = fit_submodel(self.base_router, X_router, best_estimators, fit_params)
        self.is_fitted_ = True
        return self
    
    def fit(self, X, y, fit_params={}):
        if self.router_training:
            splitted, X_router, y_router = self.split(X, y)
            return self.routed_fit(X_router, y_router, splitted, fit_params)
        # If not in router mode
        splitted = self.split(X, y)
        if self.mixed_training:
            return self.mixed_fit(splitted, fit_params)
        return self.default_fit(splitted, fit_params)

    def routed_predict(self, X):
        estimators_indices = self.estimators['router'].predict(X)

        y_pred = np.zeros((X.shape[0],1))
        estimator_track = np.zeros((X.shape[0],1), dtype='<U7')
        for idx in self.router_mapping.keys():
            mask = (estimators_indices == idx)
            if np.sum(mask) >= 1:
                X_submodel = X[mask]
                key = self.router_mapping[idx]
                y_pred[mask] = self.estimators[key].predict(X_submodel).reshape(-1,1)
                estimator_track[mask] = np.array([key] * np.sum(mask)).reshape(-1,1)
        return y_pred, estimator_track
    
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
        if self.router_training:
            y_pred, estimator_track = self.routed_predict(X)
            y_oracle = None
        else:
            strat_masks, y_oracle = self.get_stratification_masks(X, 
                                                                return_y_oracle=return_y_oracle)
            y_oracle = y_oracle.ravel() if y_oracle is not None else y_oracle

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
            y_pred = y_pred.ravel() if y_pred is not None else y_pred
        # print('\nY_pred', y_pred)
        # print("Estimator track", estimator_track)
        if y_oracle is not None:
            if return_used_estimators:
                return y_pred, y_oracle, return_used_estimators
            return y_pred, y_oracle
        if return_used_estimators:
            return y_pred, return_used_estimators
        return y_pred

    def filtered_predict(self, X, 
                         y_class: str = None,
                         pipeline = None,
                         return_mask: bool = False,
                        return_y_oracle: bool = False,
                        return_used_estimators: bool = False):
        """Retrieve only the prediction of the specified submodel class"""
        if pipeline is not None:
            X = pipeline.transform(X)
        if self.router_training:
            y_oracle = None
            if y_class == 'oracle':
                y_class = 'class1'
            estimators_indices = self.estimators['router'].predict(X)
            mask = (estimators_indices == y_class)
            y_pred = np.zeros((X.shape[0],1))
            estimator_track = np.zeros((X.shape[0],1), dtype='<U7')
            y_pred[mask] = self.estimators[y_class].predict(X[mask]).reshape(-1,1)
            y_pred = y_pred[mask]
            estimator_track[mask] = np.array([y_class] * np.sum(mask)).reshape(-1,1)
            estimator_track = estimator_track[mask]
        else:
            if y_class == 'oracle':
                strat_masks, y_oracle = self.get_stratification_masks(X, 
                                                                return_y_oracle=True)
                y_oracle = y_oracle.ravel() if y_oracle is not None else y_oracle
                mask = y_oracle > -1 # mask full of true
                y_pred = y_oracle
                estimator_track = np.zeros((X.shape[0],1), dtype='<U7')
                estimator_track = [y_class] * X.shape[0]
            else:
                strat_masks, y_oracle = self.get_stratification_masks(X, 
                                                                return_y_oracle=return_y_oracle)
                y_oracle = y_oracle.ravel() if y_oracle is not None else y_oracle
                if y_class in strat_masks.keys():
                    mask = strat_masks[y_class]
                    # print("\nMASK", mask, mask.shape)
                    # y_pred = np.zeros((X[mask].shape[0],1))
                    y_pred = np.zeros((X.shape[0],1))
                    # print("\nYPRED", y_pred, y_pred.shape)
                    # estimator_track = np.zeros((X[mask].shape[0],1), dtype='<U7')
                    estimator_track = np.zeros((X.shape[0],1), dtype='<U7')
                    y_pred[mask] = self.estimators[y_class].predict(X[mask]).reshape(-1,1)
                    y_pred = y_pred[mask]
                    estimator_track[mask] = np.array([y_class] * np.sum(mask)).reshape(-1,1)
                    estimator_track = estimator_track[mask]
                else:
                    if return_mask:
                        return None, None
                    return None

        y_pred = y_pred.ravel() if y_pred is not None else y_pred
        # Horrible code but that should do it
        if y_oracle is not None:
            if y_class == 'oracle':
                if return_used_estimators:
                    if return_mask:
                        return y_pred, return_used_estimators, mask
                    return y_pred, return_used_estimators
                if return_mask:
                    return y_pred, mask
                return y_pred

            if return_used_estimators:
                if return_mask:
                    return y_pred, y_oracle, return_used_estimators, mask
                return y_pred, y_oracle, return_used_estimators
            if return_mask:
                return y_pred, y_oracle, mask
            return y_pred, y_oracle
        
        # If no y_oracle was produced
        if return_used_estimators:
            if return_mask:
                return y_pred, return_used_estimators, mask
            return y_pred, return_used_estimators
        if return_mask:
            return y_pred, mask
        return y_pred
    
    def __sklearn_is_fitted__(self):
        return self.is_fitted_
    @property
    def booster_(self):
      return BoosterWrapper(self.estimators)
        

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