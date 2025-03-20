import pandas as pd
import numpy as np
import pickle as pkl
import json, os, gc
import warnings
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
    random_state=62,
    save=False,
    save_path="./Results/models/",
):
    if save:
        os.makedirs(save_path, exist_ok=True)
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
        "n_samples": yhat.shape[0],
    }
    # print(results)
    df = pd.DataFrame(results)

    if mode != "feature_selection":
        if save:
            # Save the model as well as permutation results.
            model_save_path = os.path.join(
                save_path, f"{estimator_name}_{ho_name}_model.txt"
            )
            estimator[-1].booster_.save_model(model_save_path)

            with open(model_save_path[-4] + "_pipeline.pkl", "wb") as f:
                pkl.dump(estimator, f)
        return df
    else:
        df["Permutation"] = ["No Permutation"]
        permutations = [df]
        input_features = list(X_train.columns)
        for feature in input_features:
            permuted_X_test = X_test.copy()

            # Apply permutation on the feature
            permuted_X_test[feature] = permuted_X_test[feature] = np.random.permutation(
                permuted_X_test[feature].values
            )

            yhat = np.ravel(estimator.predict(permuted_X_test))

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
                "n_samples": yhat.shape[0],
                "Permutation": [feature],
            }
            # print(results)
            df = pd.DataFrame(results)
            permutations.append(df)
        if save:
            # Save the model as well as permutation results.
            model_save_path = os.path.join(
                save_path, f"{estimator_name}_{ho_name}_model.txt"
            )
            estimator.booster_.save_model(model_save_path)

            with open(model_save_path[-4] + "_pipeline.pkl", "wb") as f:
                pkl.dump(estimator, f)
        return pd.concat(permutations)


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
    random_state=62,
    save=False,
    save_path="./Results/models/",
):
    feature_selection = "feature_selection" if feature_selection else "classic"
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
            mode=feature_selection,
            shuffle=shuffle,
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
    random_state=62,
    save=False,
    save_path="./Results/models/",
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
                feature_selection=feature_selection,
                shuffle=shuffle,
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
    # Check that candidates have been provided.
    assert candidates != [None], (
        "You must specify feature candidates for feature selection"
    )

    os.makedirs(save_path, exist_ok=True)
    step_list = []

    if selection_strategy == "permutation":
        # Build a pipeline that includes feature permutation for importance computation.
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
        )
        results_df = (
            results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
        )

        # -------------------------------
        # PERMUTATION FEATURE IMPORTANCE
        # -------------------------------
        # For each group defined by [Evaluation, Model, Method]:
        #   - Identify the baseline performance (where Permutation == "No Permutation").
        #   - For each candidate feature (where Permutation != "No Permutation"), compute:
        #       diff_RMSE = RMSE(permuted) - RMSE(baseline)
        #       diff_MAE  = MAE(permuted)  - MAE(baseline)
        # Then, compute the weighted RMSE and weighted MAE per candidate feature,
        # and finally compute the weighted cross mean as the average of the two.
        diff_list = []
        group_cols = ["Evaluation", "Model", "Method"]
        for name, group in results_df.groupby(group_cols):
            # Get the baseline row (no permutation) for this group.
            baseline = group[group["Permutation"] == "No Permutation"]
            if baseline.empty:
                continue
            baseline_row = baseline.iloc[0]
            baseline_rmse = baseline_row["RMSE"]
            baseline_mae = baseline_row["MAE"]

            # Process rows with a candidate feature permutation.
            permuted = group[group["Permutation"] != "No Permutation"].copy()
            if permuted.empty:
                continue
            permuted["diff_RMSE"] = permuted["RMSE"] - baseline_rmse
            permuted["diff_MAE"] = permuted["MAE"] - baseline_mae
            permuted["Weighted diff_RMSE"] = (
                permuted["diff_RMSE"]
                * permuted["n_samples"]
                / np.sum(permuted["n_samples"])
            )
            permuted["Weighted diff_MAE"] = (
                permuted["diff_MAE"]
                * permuted["n_samples"]
                / np.sum(permuted["n_samples"])
            )
            permuted["Weighted Cross Mean"] = 0.5 * (
                permuted["Weighted diff_MAE"] + permuted["Weighted diff_RMSE"]
            )
            diff_list.append(permuted)

        if len(diff_list) == 0:
            print("No permutation differences computed.")
            return None, None

        diff_df = pd.concat(diff_list, axis=0)

        # Compute weighted RMSE and weighted MAE per candidate feature.
        summary_perm = (
            diff_df.groupby("Permutation")
            .apply(
                lambda g: pd.Series(
                    {
                        "Weighted RMSE": np.sum(g["Weighted diff_RMSE"]),
                        "Weighted MAE": np.sum(g["Weighted diff_MAE"]),
                        "Unweighted RMSE": g["diff_RMSE"].mean(),
                        "Unweighted MAE": g["diff_MAE"].mean(),
                    }
                )
            )
            .reset_index()
        )

        # Compute weighted cross mean as the average of weighted RMSE and weighted MAE.
        summary_perm["Weighted Cross Mean"] = (
            summary_perm["Weighted RMSE"] + summary_perm["Weighted MAE"]
        ) / 2

        # Select the best permutation candidate based on the lowest weighted cross mean.
        best_perm_idx = summary_perm["Weighted Cross Mean"].idxmin()
        best_permutation = summary_perm.loc[best_perm_idx, "Permutation"]
        best_perm_score = summary_perm.loc[best_perm_idx, "Weighted Cross Mean"]

        print("Best permutation feature (weighted cross mean):", best_permutation)
        print("Best weighted cross mean metric:", best_perm_score)

        # Save the detailed differences and summary CSV files.
        diff_df.to_csv(
            os.path.join(
                save_path,
                f"{step_name}_{estimator_name}_{mode}_permutation_details.csv",
            ),
            index=False,
        )
        summary_perm.to_csv(
            os.path.join(
                save_path,
                f"{step_name}_summary_{estimator_name}_{mode}_permutation_results.csv",
            ),
            index=False,
        )

        return best_perm_score, best_permutation

    elif selection_strategy == "lofo":  # Leave One Feature Out
        for feature in candidates:
            remove_cols_copy = remove_cols + [feature]
            num_cols_copy = deepcopy(num_cols)
            cat_cols_copy = deepcopy(cat_cols)

            if feature in num_cols:
                num_cols_copy.remove(feature)
            if feature in cat_cols:
                cat_cols_copy.remove(feature)

            print(f"Training without {feature}..")
            print(f"Num_cols: {num_cols_copy}")
            print(f"Cat_cols: {cat_cols_copy}")

            fe_estimator = create_pipeline(
                num_cols_copy,
                cat_cols_copy,
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
                target=target,
                remove_cols=remove_cols_copy,
                random_state=random_state,
                shuffle=shuffle,
            )
            results_df = (
                results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
            )
            results_df["Removed"] = [
                f"(-) {feature}" for i in range(results_df.shape[0])
            ]
            step_list.append(results_df)

            # Remove no longer needed variables to help memory usage.
            del (
                results_df,
                results,
                fe_estimator,
                remove_cols_copy,
                num_cols_copy,
                cat_cols_copy,
            )
            gc.collect()

        step_df = pd.concat(step_list, axis=0)
        # step_df["Cross Mean (RMSE and MAE)"] = np.mean(step_df[["RMSE", "MAE"]], axis=1)

        # Compute the weighted cross mean per group using "n_samples" as weights.
        summary_step = (
            step_df.groupby("Removed")
            .apply(
                lambda g: np.sum(g["RMSE"] * g["n_samples"]) / np.sum(g["n_samples"])
            )
            .reset_index(name="Weighted RMSE")
        )

        # Compute the unweighted mean.
        summary_step["Unweighted RMSE"] = (
            step_df.groupby("Removed")["RMSE"].mean().values
        )

        # Select the best ablation based on the weighted metric.
        best_ablation = summary_step["Removed"].iloc[
            summary_step["Weighted RMSE"].idxmin()
        ]
        best_ablation_score = summary_step["Weighted RMSE"].min()

        print(
            "Best ablation feature (weighted):", best_ablation[4:]
        )  # Removing "(-) " prefix
        print("Best weighted metric:", best_ablation_score)

        step_df.to_csv(
            os.path.join(save_path, f"{step_name}_{estimator_name}_{mode}_results.csv"),
            index=False,
        )
        summary_step.to_csv(
            os.path.join(
                save_path, f"{step_name}_summary_{estimator_name}_{mode}_results.csv"
            ),
            index=False,
        )

        return best_ablation_score, best_ablation[4:]
