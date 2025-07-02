from numpy.ma.core import masked_less
import pandas as pd
import numpy as np
import pickle as pkl
import json, os, gc, subprocess, sys, time
import warnings
from typing import Union, Tuple, Any

from copy import deepcopy
from scipy import stats

from tqdm import tqdm

from datasets import (
    get_train_test_split,
    all_possible_hold_outs,
    get_hold_out_sets,
    make_feature_engineered_dataset,
)
from utils import is_gpu_available
from models import NaNFilter, StratifiedRegressor

from cuml.neighbors import KNeighborsRegressor
from cuml.ensemble import RandomForestRegressor as gpuRandomForestRegressor


from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

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
from sklearn.model_selection import train_test_split
from mapie.regression import SplitConformalRegressor

from joblib import Memory, Parallel, delayed

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
    scaler_nan_filter=False,
):
    # Mapping for numerical imputers
    if is_gpu_available():
        knn_imputer = IterativeImputer(estimator=KNeighborsRegressor(), max_iter=5)
        rf_imputer = IterativeImputer(estimator=gpuRandomForestRegressor(), max_iter=5)
    else:
        knn_imputer = KNNImputer()
        rf_imputer = IterativeImputer(estimator=RandomForestRegressor())
    # Add LinearRegressorImputer with max_iter=5.
    linear_regressor_imputer = IterativeImputer(
        estimator=LinearRegression(), max_iter=5
    )

    imputer_map = {
        "MeanImputer": SimpleImputer(strategy="mean"),
        "MedianImputer": SimpleImputer(strategy="median"),
        "KNNImputer": knn_imputer,
        "RandomForestImputer": rf_imputer,
        "LinearRegressorImputer": linear_regressor_imputer,
    }

    if scaler_nan_filter and imputer is None:
        scaler_map = {
            "StandardScaler": NaNFilter(StandardScaler()),
            "MinMaxScaler": NaNFilter(MinMaxScaler()),
            "RobustScaler": NaNFilter(RobustScaler()),  # Uses median and IQR
            "MaxAbsScaler": NaNFilter(MaxAbsScaler()),
        }
        # Pipeline for numerical features
        if scaler is not None:
            num_pipeline = Pipeline([("scaler", scaler_map[scaler])])
        else:
            num_pipeline = None
    else:
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

    transformer_pipe = [("cat", cat_pipeline, cat_cols)] if num_pipeline is None else [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    # Combine both pipelines using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformer_pipe,
        verbose_feature_names_out=False,
    )

    filter = ColumnTransformer(
        transformers=[("column_filter", "passthrough", num_cols + cat_cols)],
        verbose_feature_names_out=False,
    )

    # Complete pipeline: first preprocess, then fit the estimator.
    pipeline = Pipeline(
        [
            ("column_filter", filter),
            ("preprocessor", preprocessor),
            (model_name, estimator),
        ]
    )

    return pipeline


def downcast_df(df):
    """
    Downcast numerical columns to lower precision and convert object columns to category.
    """
    import numpy as np

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "int64":
            df[col] = df[col].astype(np.int32)
        elif col_type == "float64":
            df[col] = df[col].astype(np.float32)
        elif col_type == "object":
            df[col] = df[col].astype("category")
    return df


# --- Helper for chunked prediction and GPU cleanup ---
def predict_in_chunks(estimator, X, chunk_size=2048, y_class: str = None, conformal=False):
    preds = []
    masks = None
    if conformal:
        intervals = []
    for start in range(0, X.shape[0], chunk_size):
        chunk = X.iloc[start : start + chunk_size]

        # If y_class is passed we assume that the estimator is a StratifiedRegressor object
        if y_class is not None:
            if isinstance(estimator, Pipeline):
                out, mask = estimator[-1].filtered_predict(chunk, 
                                                      y_class=y_class, 
                                                      return_mask=True,
                                                      pipeline=estimator[:-1])
            else:
                out, mask = estimator.filtered_predict(chunk, return_mask=True,
                                                  y_class=y_class)
            if out is not None:
                preds.append(out.ravel())
            if mask is None:
                mask = np.array([False] * chunk.shape[0]).ravel()
            
            if masks is None:
                masks = [mask]
            else:
                masks.append(mask)
            
        else:
            if conformal:
                yhat, y_intervals = estimator.predict_interval(chunk)
                preds.append(yhat)
                intervals.append(y_intervals)
            else:
                preds.append(estimator.predict(chunk))
    # print(preds)
    if masks is not None:
        try:
            if conformal:
                return np.concatenate(preds).ravel(), np.concatenate(intervals), np.concatenate(masks).ravel()
            return np.concatenate(preds).ravel(), np.concatenate(masks).ravel()
        except:
            if conformal:
                return None, None, np.concatenate(masks).ravel()
            return None, np.concatenate(masks).ravel()
    try:
        if conformal:
            return np.concatenate(preds).ravel(), np.concatenate(intervals), masks
        return np.concatenate(preds).ravel(), masks
    except:
        if conformal:
            return None, None, masks
        return None, masks


def gpu_cleanup():
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"[DEBUG] GPU memory cleanup failed: {e}")


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
    mode="normal",
    y_class: str = None,
    conformal: bool = False,
    inference: bool = False,
    random_state=62,
    save=False,
    save_path="./Results/models/",
):
    import os, gc, pickle as pkl
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    # Enforce thread control for CPU libraries.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if save:
        os.makedirs(save_path, exist_ok=True)

    if hasattr(estimator, "get_params"):
        final_estimator = (
            estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        )
        params = final_estimator.get_params()

    X_train, X_test, y_train, y_test = get_train_test_split(
        ho_name,
        method_df,
        ho_sets,
        target=target,
        remove_cols=remove_cols,
        shuffle=shuffle,
        random_state=random_state,
    )
    
    if conformal:
        X_train, X_conformalize, y_train, y_conformalize = train_test_split(X_train, y_train, 
                                                                            test_size=0.2, 
                                                                            shuffle=True, 
                                                                            random_state=random_state)
        X_train = estimator[:-1].transform(X_train)
        X_conformalize = estimator[:-1].transform(X_conformalize)
        estimator = SplitConformalRegressor(
            estimator=estimator[:-1], confidence_level=0.95, conformity_score="residual_normalized")

    if not inference:
        estimator.fit(X_train, np.ravel(y_train))
        gpu_cleanup()

    if conformal:
        estimator.conformalize(X_conformalize, y_conformalize)
        yhat, yhat_intervals, mask = predict_in_chunks(estimator, X_test, y_class=y_class, conformal=conformal)
        widths = np.abs(yhat_intervals[:,1] - yhat_intervals[:,0])
        coverage = np.where((yhat >= yhat_intervals[:,0]) & (yhat <= yhat_intervals[:,1]), 1, 0).mean()
    else:
        yhat, mask = predict_in_chunks(estimator, X_test, y_class=y_class)
    gpu_cleanup()

    if mask is not None:
      y_test = y_test[mask]

    y_test_arr = y_test.to_numpy()
    if yhat is not None:
        mae = mean_absolute_error(y_test, yhat)
        try:
            rmse = root_mean_squared_error(y_test, yhat)
        except Exception as e:
            rmse = np.nan
    else:
        mae = np.nan
        rmse = np.nan

    n_samples = y_test.shape[0]    
    results = {
        "Evaluation": [ho_name],
        "Method": [method_name],
        "Model": [estimator_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "Y_hat": [yhat],
        "Y_true": [y_test_arr],
        "n_samples": [n_samples],
    }
    if conformal:
        results["Y_hat_intervals"] = [yhat_intervals.reshape(1,-1)]
        results["Width"] = [widths.reshape(1,-1)]
        results["Coverage"] = [coverage]

    df = pd.DataFrame(results)
    # Force object dtype to avoid conversion to strings
    for col in ["Y_hat_intervals", "Width"]:
        if col in df.columns:
            df[col] = df[col].astype(object)
    
    if mode != "feature_selection":
        if save:
            model_save_path = os.path.join(
                save_path, f"{estimator_name}_{ho_name}_model.txt"
            )
            estimator[-1].booster_.save_model(model_save_path)
            with open(model_save_path[:-4] + "_pipeline.pkl", "wb") as f:
                pkl.dump(estimator, f)
        return df
    else:
        df["Permutation"] = ["No Permutation"]
        permutations = [df]
        input_features = list(X_train.columns)
        for feature in input_features:
            permuted_X_test = X_test.copy(deep=False)
            permuted_X_test[feature] = X_test[feature].values[
                np.random.permutation(X_test.shape[0])
            ]

            yhat_perm = predict_in_chunks(estimator, permuted_X_test)
            gpu_cleanup()

            mae_perm = mean_absolute_error(y_test, yhat_perm)
            try:
                rmse_perm = root_mean_squared_error(y_test, yhat_perm)
            except Exception as e:
                rmse_perm = np.nan

            results_perm = {
                "Evaluation": [ho_name],
                "Method": [method_name],
                "Model": [estimator_name],
                "MAE": [mae_perm],
                "RMSE": [rmse_perm],
                "Y_hat": [yhat_perm],
                "Y_true": [y_test_arr],
                "n_samples": [yhat_perm.shape[0]],
                "Permutation": [feature],
            }
            permutations.append(pd.DataFrame(results_perm))
        if save:
            model_save_path = os.path.join(
                save_path, f"{estimator_name}_{ho_name}_model.txt"
            )
            estimator[-1].booster_.save_model(model_save_path)
            with open(model_save_path[:-4] + "_pipeline.pkl", "wb") as f:
                pkl.dump(estimator, f)
        return pd.concat(permutations)


def evaluate_method_disk_batched(
    estimator,
    estimator_name,
    method_df,
    method_name,
    ho_sets,
    target=["Score"],
    mode="controled_homology",
    feature_selection=False,
    remove_cols=[None],
    shuffle=False,
    ho_list=None,
    y_class = None,
    conformal: bool = False,
    inference: bool = False,
    random_state=62,
    save=False,
    save_path="./Results/models/",
    n_jobs_outer=8,  # Number of parallel fold evaluations for normal folds
    n_jobs_model=1,  # Number of cores per model
    batch_size=12,  # Process 12 folds at a time for normal folds
    temp_folder="./temp_results",  # Folder for intermediate results
):
    import os
    import gc
    from joblib import Parallel, delayed
    from sklearn.pipeline import Pipeline

    os.makedirs(temp_folder, exist_ok=True)

    # Set mode for feature selection if applicable.
    feature_selection = "feature_selection" if feature_selection else "classic"

    # Define heavy folds by name.
    heavy_folds = {"E.ce", "E.co", "S.en", "S.au"}

    # Separate heavy and normal folds.
    if ho_list is None:
        all_ho_names = list(ho_sets.keys())
    else:
        all_ho_names = ho_list
    batch_files = []

    # --- Process folds in parallel using disk batching ---
    if all_ho_names:
        total_normal = len(all_ho_names)
        for batch_idx, batch_start in enumerate(range(0, total_normal, batch_size)):
            print(
                f"Processing normal batch {batch_idx + 1}/{(total_normal + batch_size - 1) // batch_size}"
            )
            batch_end = min(batch_start + batch_size, total_normal)
            batch = all_ho_names[batch_start:batch_end]

            # If using LightGBM, ensure n_jobs is set appropriately in the model
            if hasattr(estimator, "n_jobs") or (
                hasattr(estimator, "get_params") and "n_jobs" in estimator.get_params()
            ):
                # For Pipeline objects, access the final estimator
                if isinstance(estimator, Pipeline):
                    final_estimator = estimator.steps[-1][1]
                    if hasattr(final_estimator, "set_params"):
                        print(f"Number of threads for each fold : {n_jobs_model}")
                        final_estimator.set_params(n_jobs=n_jobs_model)
                else:
                    print(f"Number of threads for each fold : {n_jobs_model}")
                    estimator.set_params(n_jobs=n_jobs_model)

            result_list = []
            if n_jobs_outer > 1:
                print(
                    f"Processing {len(batch)} folds in parallel with {n_jobs_outer} jobs."
                )
                result_list = Parallel(n_jobs=n_jobs_outer)(
                    delayed(evaluate_hold_out)(
                        estimator,
                        estimator_name,
                        method_df,
                        method_name,
                        ho_name,
                        ho_sets,
                        target=target,
                        remove_cols=remove_cols,
                        mode=feature_selection,
                        shuffle=shuffle,
                        y_class=y_class,
                        conformal=conformal,
                        inference=inference,
                        random_state=random_state,
                        save=save,
                        save_path=save_path,
                    )
                    for ho_name in batch
                )
            else:
                for ho_name in batch:
                    ho_df = evaluate_hold_out(
                        estimator,
                        estimator_name,
                        method_df,
                        method_name,
                        ho_name,
                        ho_sets,
                        target=target,
                        remove_cols=remove_cols,
                        mode=feature_selection,
                        shuffle=shuffle,
                        y_class=y_class,
                        conformal=conformal,
                        inference=inference,
                        random_state=random_state,
                        save=save,
                        save_path=save_path,
                    )
                    result_list.append(ho_df)

            if result_list:
                batch_df = pd.concat(result_list, axis=0)
                batch_file = os.path.join(
                    temp_folder,
                    f"batch_{batch_idx:03d}_{method_name}_{estimator_name}.csv",
                )
                batch_df.to_csv(batch_file)
                batch_files.append(batch_file)

                del result_list, batch_df
                gc.collect()

    # --- Combine all batch files ---
    print(f"Combining {len(batch_files)} batch results...")
    final_dfs = []
    for batch_file in batch_files:
        batch_df = pd.read_csv(batch_file)
        final_dfs.append(batch_df)
        del batch_df
        gc.collect()
    result_df = pd.concat(final_dfs, axis=0)

    # Optionally clean up temporary files.
    if os.environ.get("KEEP_TEMP_FILES", "0") != "1":
        for file in batch_files:
            if os.path.exists(file):
                os.remove(file)

    return result_df


def evaluate_method(
    estimator,
    estimator_name,
    method_df,
    method_name,
    ho_sets,
    target=["Score"],
    mode="controled_homology",
    feature_selection=False,
    remove_cols=[None],
    shuffle=False,
    ho_list=None,
    y_class = None,
    conformal: bool = False,
    inference: bool = False,
    random_state=62,
    save=False,
    save_path="./Results/models/",
):
    feature_selection = "feature_selection" if feature_selection else "classic"
    if ho_list is None:
        ho_list = list(ho_sets.keys())

    result_list = []
    for i, ho_name in tqdm(enumerate(ho_list)):
        ho_df = evaluate_hold_out(
            estimator,
            estimator_name,
            method_df,
            method_name,
            ho_name,
            ho_sets,
            target=target,
            remove_cols=remove_cols,
            mode=feature_selection,
            shuffle=shuffle,
            y_class=y_class,
            conformal=conformal,
            inference=inference,
            random_state=random_state,
            save=save,
            save_path=save_path,
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
    feature_selection=False,
    target=["Score"],
    remove_cols=[None],
    shuffle=False,
    ho_list=None,
    y_class = None,
    conformal=False,
    inference: bool = False,
    random_state=62,
    save=False,
    save_path="./Results/models/",
    parallel=False,
    n_jobs_outer=8,
    n_jobs_model=1,
    batch_size=12,
    temp_folder="./temp_results",
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
            # method_df = reduce_mem_usage(method_df)
            if parallel:
                results_df = evaluate_method_disk_batched(
                    estimator,
                    estimator_name,
                    method_df,
                    method_name,
                    ho_sets,
                    target=target,
                    mode=mode,
                    feature_selection=feature_selection,
                    remove_cols=remove_cols,
                    shuffle=shuffle,
                    ho_list=ho_list,
                    y_class=y_class,
                    conformal=conformal,
                    inference=inference,
                    random_state=random_state,
                    save=save,
                    save_path=save_path,
                    n_jobs_outer=n_jobs_outer,  # Number of parallel fold evaluations
                    n_jobs_model=n_jobs_model,  # Number of cores per model
                    batch_size=batch_size,  # Process 12 folds at a time
                    temp_folder=temp_folder,  # Folder for intermediate results
                )
            else:
                results_df = evaluate_method(
                    estimator,
                    estimator_name,
                    method_df,
                    method_name,
                    ho_sets,
                    target=target,
                    remove_cols=remove_cols,
                    mode=mode,
                    feature_selection=feature_selection,
                    shuffle=shuffle,
                    ho_list=ho_list,
                    y_class=y_class,
                    conformal=conformal,
                    inference=inference,
                    random_state=random_state,
                    save=save,
                    save_path=save_path,
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
    selection_strategy="permutation",
    ho_folder_path="Data/Datasets",
    suffix="_hold_outs.pkl",
    mode="permutation",
    tol=1e-3,
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
    parallel=False,
    n_jobs_outer=8,
    n_jobs_model=1,
    batch_size=12,
    temp_folder="./temp_results",
):
    # Ensure candidate features are provided.
    assert candidates != [None], (
        "You must specify feature candidates for feature selection"
    )

    os.makedirs(save_path, exist_ok=True)
    if selection_strategy == "permutation":
        # Build a pipeline that incorporates feature permutation for importance computation.
        fe_estimator = create_pipeline(
            num_cols,
            cat_cols,
            imputer=imputer,
            scaler=scaler,
            estimator=estimator,
            model_name=estimator_name,
        )

        results = evaluate(
            fe_estimator,
            estimator_name,
            dataset_dict,
            ho_folder_path=ho_folder_path,
            suffix=suffix,
            mode=mode,
            feature_selection=True,
            target=target,
            remove_cols=remove_cols,
            random_state=random_state,
            shuffle=shuffle,
            parallel=parallel,
            n_jobs_outer=n_jobs_outer,
            n_jobs_model=n_jobs_model,
            batch_size=batch_size,
            temp_folder=temp_folder,
        )
        results_df = (
            results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
        )

        # PERMUTATION FEATURE IMPORTANCE
        diff_list = []
        group_cols = ["Evaluation", "Model", "Method"]

        # Retrieve cross mean metric for the baseline.
        base_df = results_df[results_df["Permutation"] == "No Permutation"]
        test_size_vector = base_df["n_samples"].copy()
        N = base_df["n_samples"].sum()
        w_rmse = (base_df["RMSE"] * test_size_vector / N).sum()
        w_mae = (base_df["MAE"] * test_size_vector / N).sum()
        cross_mean = 0.5 * (w_rmse + w_mae)

        weight_dict = {
            ho: (base_df[base_df["Evaluation"] == ho]["n_samples"] / N).iloc[0]
            for ho in pd.unique(results_df["Evaluation"])
        }

        for name, group in results_df.groupby(group_cols):
            baseline = group[group["Permutation"] == "No Permutation"]
            if baseline.empty:
                continue
            baseline_row = baseline.iloc[0]
            baseline_rmse = baseline_row["RMSE"]
            baseline_mae = baseline_row["MAE"]

            permuted = group[group["Permutation"] != "No Permutation"].copy()
            if permuted.empty:
                continue

            permuted["diff_RMSE"] = permuted["RMSE"] - baseline_rmse
            permuted["diff_MAE"] = permuted["MAE"] - baseline_mae

            eval_key = pd.unique(group["Evaluation"])[0]
            weight = weight_dict[eval_key]

            permuted["Weighted diff_RMSE"] = (permuted["RMSE"] - baseline_rmse) * weight
            permuted["Weighted diff_MAE"] = (permuted["MAE"] - baseline_mae) * weight

            permuted["Weighted RMSE"] = permuted["RMSE"] * weight
            permuted["Weighted MAE"] = permuted["MAE"] * weight

            permuted["Weighted Cross Mean"] = 0.5 * (
                permuted["Weighted diff_RMSE"] + permuted["Weighted diff_MAE"]
            )
            diff_list.append(permuted)

        if len(diff_list) == 0:
            return None, None, None

        diff_df = pd.concat(diff_list, axis=0)

        summary_perm = (
            diff_df.groupby("Permutation")
            .apply(
                lambda g: pd.Series(
                    {
                        "Weighted diff_RMSE": np.sum(g["Weighted diff_RMSE"]),
                        "Weighted diff_MAE": np.sum(g["Weighted diff_MAE"]),
                        "Weighted RMSE": np.sum(g["Weighted RMSE"]),
                        "Weighted MAE": np.sum(g["Weighted MAE"]),
                        "Unweighted diff_RMSE": g["diff_RMSE"].mean(),
                        "Unweighted diff_MAE": g["diff_MAE"].mean(),
                    }
                )
            )
            .reset_index()
        )

        summary_perm["Weighted Cross Mean"] = (
            summary_perm["Weighted RMSE"] + summary_perm["Weighted MAE"]
        ) / 2
        summary_perm["Weighted diff Cross Mean"] = (
            summary_perm["Weighted diff_RMSE"] + summary_perm["Weighted diff_MAE"]
        ) / 2

        # best_idx = summary_perm["Weighted diff Cross Mean"].idxmin()
        # feature_to_remove = summary_perm.loc[best_idx, "Permutation"]
        # pfi = summary_perm["Weighted diff Cross Mean"].loc[best_idx]

        mask = summary_perm["Weighted diff Cross Mean"] < tol

        if np.sum(mask) > 0:
            features_to_remove = summary_perm[mask]["Permutation"]
            pfis = list(summary_perm[mask]["Weighted diff Cross Mean"])
        else:
            best_idx = summary_perm["Weighted diff Cross Mean"].idxmin()
            features_to_remove = [summary_perm.loc[best_idx, "Permutation"]]
            pfis = [summary_perm["Weighted diff Cross Mean"].loc[best_idx]]
        remove_dict = {feature: pfis[i] for i, feature in enumerate(features_to_remove)}

        details_path = os.path.join(
            save_path, f"{step_name}_{estimator_name}_{mode}_permutation_details.csv"
        )
        summary_path = os.path.join(
            save_path,
            f"{step_name}_summary_{estimator_name}_{mode}_permutation_results.csv",
        )
        print("Saving permutation details to:", details_path)
        diff_df.to_csv(details_path, index=False)
        print("Saving permutation summary to:", summary_path)
        summary_perm.to_csv(summary_path, index=False)

        return remove_dict, cross_mean

    # elif selection_strategy == "lofo":  # Leave One Feature Out
    #     for feature in candidates:
    #         remove_cols_copy = remove_cols + [feature]
    #         num_cols_copy = deepcopy(num_cols)
    #         cat_cols_copy = deepcopy(cat_cols)

    #         if feature in num_cols:
    #             num_cols_copy.remove(feature)
    #         if feature in cat_cols:
    #             cat_cols_copy.remove(feature)

    #         print(f"Training without {feature}..")
    #         print(f"Num_cols: {num_cols_copy}")
    #         print(f"Cat_cols: {cat_cols_copy}")

    #         fe_estimator = create_pipeline(
    #             num_cols_copy,
    #             cat_cols_copy,
    #             imputer=imputer,
    #             scaler=scaler,
    #             estimator=estimator,
    #             model_name=estimator_name,
    #         )

    #         results = evaluate(
    #             fe_estimator,
    #             estimator_name,
    #             dataset_dict,
    #             ho_folder_path=ho_folder_path,
    #             suffix=suffix,
    #             mode=mode,
    #             target=target,
    #             remove_cols=remove_cols_copy,
    #             random_state=random_state,
    #             shuffle=shuffle,
    #         )
    #         results_df = (
    #             results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    #         )
    #         results_df["Removed"] = [
    #             f"(-) {feature}" for i in range(results_df.shape[0])
    #         ]
    #         step_list.append(results_df)

    #         # Remove no longer needed variables to help memory usage.
    #         del (
    #             results_df,
    #             results,
    #             fe_estimator,
    #             remove_cols_copy,
    #             num_cols_copy,
    #             cat_cols_copy,
    #         )
    #         gc.collect()

    #     step_df = pd.concat(step_list, axis=0)
    #     # step_df["Cross Mean (RMSE and MAE)"] = np.mean(step_df[["RMSE", "MAE"]], axis=1)

    #     # Compute the weighted cross mean per group using "n_samples" as weights.
    #     summary_step = (
    #         step_df.groupby("Removed")
    #         .apply(
    #             lambda g: np.sum(g["RMSE"] * g["n_samples"]) / np.sum(g["n_samples"])
    #         )
    #         .reset_index(name="Weighted RMSE")
    #     )

    #     # Compute the unweighted mean.
    #     summary_step["Unweighted RMSE"] = (
    #         step_df.groupby("Removed")["RMSE"].mean().values
    #     )

    #     # Select the best ablation based on the weighted metric.
    #     best_ablation = summary_step["Removed"].iloc[
    #         summary_step["Weighted RMSE"].idxmin()
    #     ]
    #     best_ablation_score = summary_step["Weighted RMSE"].min()

    #     print(
    #         "Best ablation feature (weighted):", best_ablation[4:]
    #     )  # Removing "(-) " prefix
    #     print("Best weighted metric:", best_ablation_score)

    #     step_df.to_csv(
    #         os.path.join(save_path, f"{step_name}_{estimator_name}_{mode}_results.csv"),
    #         index=False,
    #     )
    #     summary_step.to_csv(
    #         os.path.join(
    #             save_path, f"{step_name}_summary_{estimator_name}_{mode}_results.csv"
    #         ),
    #         index=False,
    #     )

    #     return best_ablation_score, best_ablation[4:]

def load_best_hyperparams(path_optuna: str = None, path_ref_score: str = None):
    if path_optuna is None:
        path_optuna = "./Results/optuna_campaign/optuna_study.pkl"
    if path_ref_score is None:
        path_ref_score = "./Results/feature_engineering/best_score.pkl"

    # Retrieve optuna campaign best params
    with open(path_optuna, "rb") as f:
        study = pkl.load(f)
        best_params = study.best_trial.params

    with open(path_ref_score, "rb") as f:
        ref_score = pkl.load(f)

    if study.best_trial.value >= ref_score:
        # default values
        print("Default yielded better results.. Using Default parameters")
        best_params = {}
    return best_params

def load_best_model(model_class, load_path_kwargs={}):
    best_params = load_best_hyperparams(**load_path_kwargs)

    best_params["force_col_wise"] = True
    best_params["random_state"] = 62
    best_params['n_jobs'] = 1
    best_params["tree_learner"] = 'serial'
    best_params["verbose_eval"] = False
    best_params['verbose'] = -1
    return model_class(**best_params)

def make_best_estimator(model_class, model_name,
                        stratified=False, 
                        stratify_params={},
                        no_imputation=False,
                        path_dataset="Data/Datasets/fe_combinatoric_COI.csv", 
                        load_path_kwargs={}):
    combinatoric_df = pd.read_csv(path_dataset)
    df_dict = {"combinatoric": combinatoric_df}

    target = ["Score"]
    cat_cols = ["Modele"]
    remove_cols = [
        "Unnamed: 0",
        "Unnamed: 0.1",
        "B_sample_ID",
        "P_sample_ID",
        "Bacillus",
        "Pathogene",
    ]
    num_cols = [
        col
        for col in df_dict["combinatoric"].columns
        if col not in cat_cols + remove_cols + target
    ]

    best_model = load_best_model(model_class, load_path_kwargs)
    if stratified:
        best_model = StratifiedRegressor(best_model, **stratify_params)

    if no_imputation:
        return create_pipeline(
                num_cols,
                cat_cols,
                imputer=None,
                scaler="RobustScaler",
                estimator=best_model,
                model_name=model_name,
                scaler_nan_filter=True,
            )
    return create_pipeline(
                num_cols,
                cat_cols,
                imputer="KNNImputer",
                scaler="RobustScaler",
                estimator=best_model,
                model_name=model_name,
            )
