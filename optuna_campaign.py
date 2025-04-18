import pandas as pd
import numpy as np
import pickle as pkl
import os, gc, re
import cupy as cp

from lightgbm import LGBMRegressor
from pipeline import create_pipeline, evaluate
import optuna

# Feature selection experiment showed that all native features were useful and none of the designed features were.
# So we can simply use our original dataset
combinatoric_df = pd.read_csv("Data/Datasets/fe_combinatoric_COI.csv")

df_dict = {"combinatoric": combinatoric_df}

target = ["Score"]
cat_cols = ["Modele"]
remove_cols = ["Unnamed: 0", 'Unnamed: 0.1', "B_sample_ID", "P_sample_ID", "Bacillus", "Pathogene"]
num_cols = [
    col
    for col in df_dict["combinatoric"].columns
    if col not in cat_cols + remove_cols + target
]

print(combinatoric_df.columns)

def objective(trial):
    param = {
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_jobs": 1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 64),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": 62,
        "gpu_use_dp": False,
        "tree_learner": "serial",
        "verbose_eval": False,
        "metric": trial.suggest_categorical(
            "metric", ["l1", "l2", "huber", "quantile"]
        ),
    }

    estimator = LGBMRegressor(**param)

    estimator = create_pipeline(
        num_cols,
        cat_cols,
        imputer="KNNImputer",
        scaler="RobustScaler",
        estimator=estimator,
        model_name="LGBMRegressor",
    )
    results = evaluate(
        estimator,
        "LGBMRegressor",
        df_dict,
        ho_folder_path="Data/Datasets/",
        suffix="_hold_outs.pkl",
        mode="ho",
        target=target,
        remove_cols=remove_cols,
        random_state=62,
        shuffle=False,
        parallel=True,
        n_jobs_outer=12,
        n_jobs_model=1,
        batch_size=12,
        temp_folder="./temp_results",
    )

    # Compute terms for the weighted average
    results["Weighted RMSE"] = (
        results["RMSE"] * results["n_samples"] / results["n_samples"].sum()
    )

    results["Weighted MAE"] = (
        results["MAE"] * results["n_samples"] / results["n_samples"].sum()
    )

    # Compute the cross mean
    cross_mean = 0.5 * (
        np.sum(results["Weighted RMSE"]) + np.sum(results["Weighted MAE"])
    )

    # Return the cross mean
    return cross_mean


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    os.makedirs("./Results/optuna_campaign/", exist_ok=True)
    with open("./Results/optuna_campaign/optuna_study.pkl", "wb") as f:
        pkl.dump(study, f)
    
