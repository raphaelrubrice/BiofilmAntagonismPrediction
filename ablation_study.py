import pandas as pd
import numpy as np
import pickle as pkl
import os
import gc
import cupy as cp

from lightgbm import LGBMRegressor
from pipeline import create_pipeline, evaluate
from plots import plot_feature_selection

if __name__ == "__main__":
    combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_selected_FE.csv")

    df_dict = {"combinatoric": combinatoric_df}

    target = ["Score"]
    cat_cols = ["Modele"]
    remove_cols = [
        "Unnamed: 0",
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

    # Retrieve optuna campaign best params:
    with open("./Results.optuna_campaign/optuna_study.pkl", "rb") as f:
        study = pkl.load(f)
        best_params = study.best_trial.params

    estimator = LGBMRegressor(**best_params)

    os.makedirs("Results/ablation_study/", exist_ok=True)
    Ablations = [None] + ["all_B", "all_P"] + [col for col in combinatoric_df.columns]
    for remove in Ablations:
        if remove == "all_B":
            remove_list = [
                col for col in combinatoric_df.columns if col.startswith("B_")
            ]
        elif remove == "all_P":
            remove_list = [
                col for col in combinatoric_df.columns if col.startswith("P_")
            ]
        elif remove is None:
            remove_list = []
        else:
            remove_list = [remove]

        num_cols_copy = list(set(num_cols).difference(set(remove_list)))
        cat_cols_copy = list(set(cat_cols).difference(set(remove_list)))

        estimator = create_pipeline(
            num_cols_copy,
            cat_cols_copy,
            imputer="KNNImputer",
            scaler="RobustScaler",
            estimator=estimator,
            model_name="LGBMRegressor",
        )
        results = evaluate(
            estimator,
            "LGBMRegressor",
            df_dict,
            mode="ho",
            suffix="_hold_outs.pkl",
            ho_folder_path="Data/Datasets/",
            target=target,
            remove_cols=remove_cols + remove_list,
        )

        results.to_csv(f"Results/ablation_study/ho_{remove}_LGBMRegressor_results.csv")

    all_results = []
    for file in os.listdir("Results/ablation_study/"):
        if file.endswith(".csv") and "ho" in file and "_all_results" not in file:
            all_results.append(pd.read_csv("Results/ablation_study/" + file))
    if all_results != []:
        all_df = pd.concat(all_results, axis=0)
        all_df.to_csv("Results/ablation_study/ho_all_results.csv")
    else:
        print("No result files found.")
