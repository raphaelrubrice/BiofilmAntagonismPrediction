from sklearn.base import BaseEstimator
from xgboost import XGBRFRegressor, XGBRegressor

class skready_XGBRFRegressor(XGBRFRegressor, BaseEstimator):
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=3, 
                 learning_rate=0.1, 
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=None):
        # Explicitly initialize the XGBoost regressor with these parameters
        XGBRegressor.__init__(self,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=random_state)
        # Also initialize BaseEstimator
        BaseEstimator.__init__(self)
        
        # Save parameters as attributes (for get_params/set_params)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def __sklearn_tags__(self):
        try:
            tags = super(skready_XGBRFRegressor, self).__sklearn_tags__()
        except AttributeError:
            tags = {}
        tags['target_tags'] = {'single_output': True}
        tags['non_deterministic'] = True
        return tags

class skready_XGBRegressor(XGBRegressor, BaseEstimator):
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=3, 
                 learning_rate=0.1, 
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=None):
        tags = {}
        tags['target_tags'] = {'single_output': True}
        tags['non_deterministic'] = True
        self.__sklearn_tags__ = tags

        # Explicitly initialize the XGBoost regressor with these parameters
        XGBRegressor.__init__(self,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=random_state)
        # Also initialize BaseEstimator
        BaseEstimator.__init__(self)
        
        # Save parameters as attributes (for get_params/set_params)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def __sklearn_tags__(self):
        try:
            tags = super(skready_XGBRegressor, self).__sklearn_tags__()
        except AttributeError:
            tags = {}
        tags['target_tags'] = {'single_output': True}
        tags['non_deterministic'] = True
        return tags