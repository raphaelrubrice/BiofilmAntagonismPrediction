import pandas as pd
import numpy as np
import json, os
import warnings

from tqdm import tqdm

from datasets import get_train_test_split, all_possible_hold_outs, get_hold_out_sets

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score

from joblib import Memory
memory = Memory(location='./cachedir', verbose=0)
set_config(transform_output="pandas")
# def create_pipeline(imputer, scaler, estimator, model_name="Regressor"):
#     imputer_map = {"MeanImputer":SimpleImputer(strategy="mean"),
#                    "MedianImputer":SimpleImputer(strategy="median"),
#                    "KNNImputer":KNNImputer(),
#                    "IterativeImputer":IterativeImputer(RandomForestRegressor())
#                     }
    
#     scaler_map = {"StandardScaler":StandardScaler(),
#                   "MinMaxScaler":MinMaxScaler(),
#                   "RobustScaler":RobustScaler(), # Median and IQR
#                   "MaxAbsScaler":MaxAbsScaler()
#                   }
    
#     if scaler is not None:
#         pipeline = Pipeline([(imputer, imputer_map[imputer]), 
#                             (scaler, scaler_map[scaler]),
#                             (model_name, estimator)])
#     else:
#         pipeline = Pipeline([(imputer, imputer_map[imputer]), 
#                             (model_name, estimator)])
#     return pipeline
bacillus_bank = ["1167", "1202", "1218", "1219", "1234", "1273", "1298", "1339", 
                 "11285", "11457", "12001", "12048", "12701", "12832", "B1", "B8", "B18", "C5"]
pathogen_bank = ["E.ce", "E.co", "S.en", "S.au"]

def create_pipeline(num_cols, cat_cols, imputer='MeanImputer', scaler='StandardScaler', estimator=None, model_name="Regressor"):
    # Mapping for numerical imputers
    imputer_map = {"MeanImputer": SimpleImputer(strategy="mean"),
                   "MedianImputer": SimpleImputer(strategy="median"),
                   "KNNImputer": KNNImputer(),
                   "IterativeImputer": IterativeImputer(estimator=RandomForestRegressor())}
    
    scaler_map = {"StandardScaler": StandardScaler(),
                  "MinMaxScaler": MinMaxScaler(),
                  "RobustScaler": RobustScaler(),  # Uses median and IQR
                  "MaxAbsScaler": MaxAbsScaler()}
    
    # Pipeline for numerical features
    if scaler is not None:
        num_pipeline = Pipeline([
            ('imputer', imputer_map[imputer]),
            ('scaler', scaler_map[scaler])
        ])
    else:
        num_pipeline = Pipeline([
            ('imputer', imputer_map[imputer])
        ])

    # Pipeline for categorical features: impute missing values and OneHotEncode
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(sparse_output=False))
    ])
    
    # Combine both pipelines using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    # Complete pipeline: first preprocess, then fit the estimator.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        (model_name, estimator),
    ])
    
    return pipeline

def evaluate_hold_out(estimator, estimator_name,
                    method_df, method_name, ho_name, ho_sets, target=["Score"], remove_cols=[None],
                      shuffle=False, random_state=62):
    X_train, X_test, y_train, y_test = get_train_test_split(ho_name, method_df, ho_sets, target=target,
                                                            remove_cols=remove_cols, 
                                                            shuffle=shuffle, random_state=random_state)

    estimator.fit(X_train, np.ravel(y_train))

    yhat = estimator.predict(X_test)
    
    abs_err = np.abs(yhat - y_test.to_numpy())
    relative_abs_err = np.abs(yhat - y_test.to_numpy()) / (np.maximum(1e-3, np.abs(y_test.to_numpy())))

    # results = {ho_name : {
    #                 "yhat":yhat.tolist(),
    #                 "y_true":y_test.to_numpy().tolist(),
    #                 "mae":mean_absolute_error(y_test, yhat),
    #                 "std_abs_err":np.std(abs_err, axis=0)[0],
    #                 "mape":mean_absolute_percentage_error(y_test, yhat),
    #                 "std_relative_abs_err":np.std(relative_abs_err, axis=0)[0],
    #                 "rmse":root_mean_squared_error(y_test, yhat),
    #                 "r2":r2_score(y_test, yhat)
    #                     }
    #         }
    results = {"Evaluation":[ho_name], "Method":[method_name], "Model":[estimator_name], 
                "MAE":[mean_absolute_error(y_test, yhat)],
                "std_abs_err":[np.std(abs_err, axis=0)[0]],
                "MAPE":[mean_absolute_percentage_error(y_test, yhat)],
                "std_relative_abs_err":[np.std(relative_abs_err, axis=0)[0]],
                "RMSE":[root_mean_squared_error(y_test, yhat)],
                "R2":[r2_score(y_test, yhat)],
                "Y_hat": [yhat],
               "Y_true":[y_test.to_numpy()]
                }
    # print(results)
    df = pd.DataFrame(results)
    return df

def evaluate_method(estimator, estimator_name, 
            method_df, method_name, ho_sets, target=["Score"], mode="controled_homology", remove_cols=[None],
            shuffle=False, random_state=62):
    result_list = []
    # if mode == "controled_homology":
    #     all_possible_ho = all_possible_hold_outs(return_names=True)
    #     # first_bacillus = None
    #     # first_pathogen = None
    #     # first_interaction = None

    #     for i, ho_name in tqdm(enumerate(all_possible_ho)):
    #         ho_df = evaluate_hold_out(estimator, estimator_name, 
    #                     ho_name, method_df, method_name, ho_sets, target=target, remove_cols=remove_cols,
    #                     shuffle=shuffle, random_state=random_state)
    #         result_list.append(ho_df)
            # ho_dict = evaluate_hold_out(estimator, estimator_name, 
            #             ho_name, method_df, method_name, ho_sets, target=target, remove_cols=remove_cols,
            #             shuffle=shuffle, random_state=random_state)
            
            # if ho_name in bacillus_bank and first_bacillus is None:
            #     first_bacillus = True
            #     method_ho_dict["overall_bacillus"] = {key:val for key, val in ho_dict[ho_name].items()}
            # elif ho_name in pathogen_bank and first_pathogen is None:
            #     first_pathogen = True
            #     method_ho_dict["overall_pathogen"] = {key:val for key, val in ho_dict[ho_name].items()}
            # elif first_interaction is None:
            #     first_interaction = True
            #     method_ho_dict["overall_interaction"] = {key:val for key, val in ho_dict[ho_name].items()}

            # if i == 0:
            #     method_ho_dict[method_name] = ho_dict
            #     method_ho_dict["overall"] = {key:val for key, val in ho_dict[ho_name].items()}
            # else:
            #     method_ho_dict[method_name].update(ho_dict)
            #     for key, val in ho_dict[ho_name].items():
            #         method_ho_dict["overall"][key] += val

            #         if ho_name in bacillus_bank:
            #             method_ho_dict["overall_bacillus"][key] += val
            #         elif ho_name in pathogen_bank:
            #             method_ho_dict["overall_pathogen"][key] += val
            #         else:
            #             method_ho_dict["overall_interaction"][key] += val

        # compute means
        # for key in method_ho_dict["overall"].keys():
        #     if key not in ["yhat", "y_true"]:
        #         method_ho_dict["overall"][key] /= len(all_possible_ho)
        #         method_ho_dict["overall_bacillus"][key] /= len(bacillus_bank)
        #         method_ho_dict["overall_pathogen"][key] /= len(pathogen_bank)
        #         method_ho_dict["overall_interaction"][key] /= (len(all_possible_ho) - (len(pathogen_bank) + len(bacillus_bank)))
    # else:
    for i, ho_name in tqdm(enumerate(ho_sets.keys())):
        ho_df = evaluate_hold_out(estimator, estimator_name,
                    method_df, method_name, ho_name, ho_sets, target=target, remove_cols=remove_cols,
                    shuffle=shuffle, random_state=random_state)
        result_list.append(ho_df)
        #     if i == 0:
        #         method_ho_dict[method_name] = ho_dict
        #         method_ho_dict["overall"] = {key:val for key, val in ho_dict[ho_name].items()}
        #     else:
        #         method_ho_dict[method_name].update(ho_dict)
        #         for key, val in ho_dict[ho_name].items():
        #             method_ho_dict["overall"][key] += val

        # for key in method_ho_dict["overall"].keys():
        #     if key not in ["yhat", "y_true"]:
        #         method_ho_dict["overall"][key] /= len(ho_sets.keys())
    result_df = pd.concat(result_list, axis=0)
    return result_df

def evaluate(estimator, estimator_name, 
            dataset_dict, ho_folder_path="Data/Datasets", suffix="_hold_outs.pkl", mode='controled_homology',
            target=["Score"], remove_cols=[None], shuffle=False, random_state=62):
    results = []
    for method_name in tqdm(["avg", "random", "combinatoric"]):
        if method_name in dataset_dict.keys():
            if "B_sample_ID" not in remove_cols and method_name == "combinatoric":
                remove_cols.append("B_sample_ID")
            if "P_sample_ID" not in remove_cols and method_name == "combinatoric":
                remove_cols.append("P_sample_ID")    

            method_df = dataset_dict[method_name]
            ho_sets = get_hold_out_sets(method_name, ho_folder_path=ho_folder_path, suffix=suffix)
            results_df = evaluate_method(estimator, estimator_name, 
                                            method_df, method_name, ho_sets, 
                                            target=target, remove_cols=remove_cols, mode=mode,
                                            shuffle=shuffle, random_state=random_state)
            results.append(results_df)
        else:
            warn_message = f"{method_name} not found in dataset_dict with keys {dataset_dict.keys()}. Skipping this method"
            warnings.warn(warn_message)
    return pd.concat(results, axis=0)