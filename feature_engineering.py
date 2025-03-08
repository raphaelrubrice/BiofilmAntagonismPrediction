import os
import pandas as pd
import numpy as np

from pipeline import create_pipeline, select_features
from datasets import make_feature_engineered_dataset
from plots import plot_feature_selection

from lightgbm import LGBMRegressor


if __name__ == "__main__":
    combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_COI.csv")

    df_dict = {"combinatoric": combinatoric_df}

    target = ["Score"]
    cat_cols = ["Modele"]
    remove_cols = ["Unnamed: 0", "B_sample_ID", "P_sample_ID", "Bacillus", "Pathogene"]
    num_cols = [
        col
        for col in df_dict["combinatoric"].columns
        if col not in cat_cols + remove_cols + target
    ]

    estimator = LGBMRegressor(
        random_state=62,
        n_jobs=-1,
        gpu_use_dp=False,
        tree_learner="serial",
        device="cuda",
        verbose_eval=False,
        verbose=-1,
    )
    estimator_name = "LGBMRegressor"

    best_ablation = None
    previous = (0.207 + 0.158) / 2  # RMSE and MAE Cross target mean value
    current = 0

    i = 0
    candidates = num_cols + cat_cols

    # Save Feature Engineering to avoid recomputation at each step
    FE_combinatoric_df = make_feature_engineered_dataset(
        combinatoric_df,
        "Data/Datasets/combinatoric_FeatureEng.csv",
        cols_prod=candidates,
        cols_ratio=num_cols,
        cols_pow=num_cols,
        pow_orders=[2, 3],
        eps=1e-4,
        target=target,
        remove_cols=remove_cols,
    )

    df_dict = {"combinatoric": FE_combinatoric_df}

    while previous > current and len(candidates) > 1:
        if i != 0:
            # Remove previously eliminated feature
            candidates.remove(best_ablation)
            if best_ablation in num_cols:
                num_cols.remove(best_ablation)
            if best_ablation in cat_cols:
                cat_cols.remove(best_ablation)

            # Add it to remove_cols
            remove_cols.append(best_ablation)
            previous = current

        current, best_ablation = select_features(
            estimator,
            estimator_name,
            df_dict,
            ho_folder_path="Data/Datasets/",
            suffix="_hold_outs.pkl",
            mode="controled_homology",
            target=["Score"],
            candidates=candidates,
            remove_cols=remove_cols,
            save_path="Results/feature_engineering",
            step_name=f"step_{i + 1}",
            shuffle=False,
            random_state=62,
            imputer="KNNImputer",
            scaler="RobustScaler",
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        i += 1

    print("*********")
    print(f"Step Best: {current}, without {best_ablation}")
    print("*********")

    plot_feature_selection(
        "Results/feature_engineering", "RMSE", "Results/feature_engineering"
    )
    plot_feature_selection(
        "Results/feature_engineering", "MAE", "Results/feature_engineering"
    )
