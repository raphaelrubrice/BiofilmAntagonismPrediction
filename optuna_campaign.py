import pandas as pd
import numpy as np
import pickle as pkl
import os, gc
import cupy as cp

from lightgbm import LGBMRegressor
from pipeline import create_pipeline, evaluate
import optuna


combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_selected_FE.csv")

df_dict = {"combinatoric": combinatoric_df}

target = ["Score"]
cat_cols = ["Modele"]
remove_cols = ["Unnamed: 0", "B_sample_ID", "P_sample_ID", "Bacillus", "Pathogene"]
num_cols = [
    col
    for col in df_dict["combinatoric"].columns
    if col not in cat_cols + remove_cols + target
]


def objective(trial):
    param = {
        "objective": "regression",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 64),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": 62,
        "gpu_use_dp": False,
        "tree_learner": "serial",
        "device": "cuda",
        "verbose_eval": False,
    }

    estimator = LGBMRegressor(**param)

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
    )
    results["Cross Mean (RMSE and MAE)"] = np.mean(results[["RMSE", "MAE"]], axis=1)
    results["Weighted Cross Mean (RMSE and MAE)"] = (
        results["Cross Mean (RMSE and MAE)"]
        * results["n_samples"]
        / results["n_samples"].sum()
    )

    return np.mean(results["Weighted Cross Mean (RMSE and MAE)"])


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    os.mkdir("./Results/optuna_campaign/", exist_ok=True)
    with open("./Results/optuna_campaign/optuna_study.pkl", "wb") as f:
        pkl.dump(study, f)
