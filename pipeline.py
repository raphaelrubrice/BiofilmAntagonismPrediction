import pandas as pd
import numpy as np
import json, os
import warnings
from copy import deepcopy

from tqdm import tqdm

from datasets import (
    get_train_test_split,
    all_possible_hold_outs,
    get_hold_out_sets,
    make_feature_engineered_dataset,
)
from utils import is_gpu_available

from cuml.neighbors import KNeighborsRegressor
from cuml.ensemble import RandomForestRegressor as gpuRandomForestRegressor

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    OneHotEncoder,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)

from joblib import Memory

memory = Memory(location="./cachedir", verbose=0)
set_config(transform_output="pandas")

bacillus_bank = [
    "1167",
    "1202",
    "1218",
    "1219",
    "1234",
    "1273",
    "1298",
    "1339",
    "11285",
    "11457",
    "12001",
    "12048",
    "12701",
    "12832",
    "B1",
    "B8",
    "B18",
    "C5",
]
pathogen_bank = ["E.ce", "E.co", "S.en", "S.au"]


def create_pipeline(
    num_cols,
    cat_cols,
    imputer="MeanImputer",
    scaler="StandardScaler",
    estimator=None,
    model_name="Regressor",
):
    # Mapping for numerical imputers
    if is_gpu_available():
        knn_imputer = IterativeImputer(estimator=KNeighborsRegressor())
        rf_imputer = IterativeImputer(estimator=gpuRandomForestRegressor())
    else:
        knn_imputer = KNNImputer()
        rf_imputer = IterativeImputer(estimator=RandomForestRegressor())
    imputer_map = {
        "MeanImputer": SimpleImputer(strategy="mean"),
        "MedianImputer": SimpleImputer(strategy="median"),
        "KNNImputer": knn_imputer,
        "RandomForestImputer": rf_imputer,
    }

    scaler_map = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),  # Uses median and IQR
        "MaxAbsScaler": MaxAbsScaler(),
    }

    # Pipeline for numerical features
    if scaler is not None:
        num_pipeline = Pipeline(
            [("imputer", imputer_map[imputer]), ("scaler", scaler_map[scaler])]
        )
    else:
        num_pipeline = Pipeline([("imputer", imputer_map[imputer])])

    # Pipeline for categorical features: impute missing values and OneHotEncode
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(sparse_output=False))])

    # Combine both pipelines using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)],
        verbose_feature_names_out=False,
    )

    # Complete pipeline: first preprocess, then fit the estimator.
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (model_name, estimator),
        ]
    )

    return pipeline


def evaluate_hold_out(
    estimator,
    estimator_name,
    method_df,
    method_name,
    ho_name,
    ho_sets,
    target=["Score"],
    remove_cols=[None],
    shuffle=False,
    random_state=62,
):
    X_train, X_test, y_train, y_test = get_train_test_split(
        ho_name,
        method_df,
        ho_sets,
        target=target,
        remove_cols=remove_cols,
        shuffle=shuffle,
        random_state=random_state,
    )

    estimator.fit(X_train, np.ravel(y_train))

    yhat = np.ravel(estimator.predict(X_test))

    abs_err = np.abs(yhat - y_test.to_numpy())
    relative_abs_err = np.abs(yhat - y_test.to_numpy()) / (
        np.maximum(1e-3, np.abs(y_test.to_numpy()))
    )

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
    results = {
        "Evaluation": [ho_name],
        "Method": [method_name],
        "Model": [estimator_name],
        "MAE": [mean_absolute_error(y_test, yhat)],
        "std_abs_err": [np.std(abs_err, axis=0)[0]],
        "MAPE": [mean_absolute_percentage_error(y_test, yhat)],
        "std_relative_abs_err": [np.std(relative_abs_err, axis=0)[0]],
        "RMSE": [root_mean_squared_error(y_test, yhat)],
        "R2": [r2_score(y_test, yhat)],
        "Y_hat": [yhat],
        "Y_true": [y_test.to_numpy()],
    }
    # print(results)
    df = pd.DataFrame(results)
    return df


def evaluate_method(
    estimator,
    estimator_name,
    method_df,
    method_name,
    ho_sets,
    target=["Score"],
    mode="controled_homology",
    remove_cols=[None],
    shuffle=False,
    random_state=62,
):
    result_list = []
    for i, ho_name in tqdm(enumerate(ho_sets.keys())):
        ho_df = evaluate_hold_out(
            estimator,
            estimator_name,
            method_df,
            method_name,
            ho_name,
            ho_sets,
            target=target,
            remove_cols=remove_cols,
            shuffle=shuffle,
            random_state=random_state,
        )
        result_list.append(ho_df)
    result_df = pd.concat(result_list, axis=0)
    return result_df


def evaluate(
    estimator,
    estimator_name,
    dataset_dict,
    ho_folder_path="Data/Datasets",
    suffix="_hold_outs.pkl",
    mode="controled_homology",
    target=["Score"],
    remove_cols=[None],
    shuffle=False,
    random_state=62,
):
    results = []
    for method_name in tqdm(["avg", "random", "combinatoric"]):
        if method_name in dataset_dict.keys():
            if "B_sample_ID" not in remove_cols and method_name == "combinatoric":
                remove_cols.append("B_sample_ID")
            if "P_sample_ID" not in remove_cols and method_name == "combinatoric":
                remove_cols.append("P_sample_ID")

            method_df = dataset_dict[method_name]
            ho_sets = get_hold_out_sets(
                method_name, ho_folder_path=ho_folder_path, suffix=suffix
            )
            results_df = evaluate_method(
                estimator,
                estimator_name,
                method_df,
                method_name,
                ho_sets,
                target=target,
                remove_cols=remove_cols,
                mode=mode,
                shuffle=shuffle,
                random_state=random_state,
            )
            results.append(results_df)
        else:
            warn_message = f"{method_name} not found in dataset_dict with keys {dataset_dict.keys()}. Skipping this method"
            warnings.warn(warn_message)
    return pd.concat(results, axis=0)


def select_features(
    estimator,
    estimator_name,
    dataset_dict,
    ho_folder_path="Data/Datasets",
    suffix="_hold_outs.pkl",
    mode="controled_homology",
    target=["Score"],
    candidates=[None],
    remove_cols=[None],
    save_path="Results/native_feature_selection",
    step_name="1st",
    shuffle=False,
    random_state=62,
    imputer="KNNImputer",
    scaler="RobustScaler",
    num_cols=None,
    cat_cols=None,
):
    assert candidates != [None], (
        "You must specify feature candidates for feature selection"
    )

    os.makedirs(save_path, exist_ok=True)

    step_list = []
    for feature in candidates:
        remove_cols_copy = remove_cols + [feature]

        if feature in num_cols:
            num_cols_copy = deepcopy(num_cols)
            num_cols_copy.remove(feature)
        if feature in cat_cols:
            cat_cols_copy = deepcopy(cat_cols)
            cat_cols_copy.remove(feature)

        estimator = create_pipeline(
            num_cols_copy,
            cat_cols_copy,
            imputer=imputer,
            scaler=scaler,
            estimator=estimator,
            model_name=estimator_name,
        )

        results = evaluate(
            estimator,
            estimator_name,
            dataset_dict,
            ho_folder_path=ho_folder_path,
            suffix=suffix,
            mode=mode,
            target=target,
            remove_cols=remove_cols_copy,
            random_state=random_state,
            shuffle=shuffle,
        )
        results_df = (
            results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
        )
        results_df["Removed"] = [f"(-) {feature}" for i in range(results_df.shape[0])]
        step_list.append(results_df)

    step_df = pd.concat(step_list, axis=0)
    step_df["Cross Mean (RMSE and MAE)"] = np.mean(step_df[["RMSE", "MAE"]], axis=1)
    best_ablation = step_df["Removed"][
        step_df["Cross Mean (RMSE and MAE)"]
        == step_df["Cross Mean (RMSE and MAE)"].min()
    ].iloc[0]
    best_ablation_score = step_df["Cross Mean (RMSE and MAE)"][
        step_df["Removed"] == best_ablation
    ].iloc[0]
    step_df.to_csv(
        save_path + f"/{step_name}_{estimator_name}_{mode}_results.csv",
        index=False,
    )

    return best_ablation_score, best_ablation[4:]
