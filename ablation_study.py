import fireducks.pandas as pd
import numpy as np
import pickle as pkl
import os
import gc
import cupy as cp

from lightgbm import LGBMRegressor
from pipeline import create_pipeline, evaluate

if __name__ == "__main__":
    # Read the dataset and create a dictionary of DataFrames
    combinatoric_df = pd.read_csv("Data/Datasets/fe_combinatoric_COI.csv")
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

    # Retrieve optuna campaign best params
    with open("./Results/optuna_campaign/optuna_study.pkl", "rb") as f:
        study = pkl.load(f)
        best_params = study.best_trial.params

    with open("./Results/feature_engineering/best_score.pkl", "rb") as f:
        ref_score = pkl.load(f)

    if study.best_trial.value >= ref_score:
        # default values
        best_params = {}

    best_params["force_col_wise"] = True
    best_params["random_state"] = 62
    best_params['n_jobs'] = 1
    best_params["tree_learner"] = 'serial'
    best_params["verbose_eval"] = False
    best_params['verbose'] = -1
    model = LGBMRegressor(**best_params)

    os.makedirs("Results/ablation_study/", exist_ok=True)
    Ablations = (
        [None]
        + ["all_B", "all_P"]
        + [col for col in combinatoric_df.columns if col not in remove_cols]
    )

    for remove in Ablations:
        if remove == "all_B":
            remove_list = [col for col in combinatoric_df.columns if "B_" in col]
        elif remove == "all_P":
            remove_list = [col for col in combinatoric_df.columns if "P_" in col]
        elif remove is None:
            remove_list = []
        else:
            remove_list = [remove]

        num_cols_copy = list(set(num_cols).difference(set(remove_list)))
        cat_cols_copy = list(set(cat_cols).difference(set(remove_list)))

        print("Numerical columns:", num_cols_copy)
        print("Categorical columns:", cat_cols_copy)
        print("Removed columns:", remove_list)

        estimator = create_pipeline(
            num_cols_copy,
            cat_cols_copy,
            imputer="KNNImputer",
            scaler="RobustScaler",
            estimator=model,
            model_name="LGBMRegressor",
        )

        save = True if remove is None else False
        results = evaluate(
            estimator,
            "LGBMRegressor",
            df_dict,
            mode="ho",
            suffix="_hold_outs.pkl",
            ho_folder_path="Data/Datasets/",
            target=target,
            remove_cols=remove_cols + remove_list,
            save=save,
            save_path="./Results/models/",
            parallel=True,
            n_jobs_outer=12,
            n_jobs_model=1,
            batch_size=12,
            temp_folder="./temp_results",
        )
        results.to_csv(f"Results/ablation_study/ho_{remove}_LGBMRegressor_results.csv")

        # Remove variables that are no longer needed to free GPU memory
        del results, estimator, num_cols_copy, cat_cols_copy, remove_list
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    all_results = []
    ablation_folder = "Results/ablation_study/"
    for file in os.listdir(ablation_folder):
        if file.endswith(".csv") and "ho" in file and "_all_results" not in file:
            all_results.append(pd.read_csv(os.path.join(ablation_folder, file)))
    if all_results:
        all_df = pd.concat(all_results, axis=0)
        all_df.to_csv("Results/ablation_study/ho_all_results.csv")
    else:
        print("No result files found.")

    # Final clean-up: remove any remaining temporary objects and free GPU memory.
    del combinatoric_df, df_dict, model, best_params, all_results
    if study:
        del study
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
