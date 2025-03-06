import pandas as pd
import numpy as np
import os
import gc
import cupy as cp

from lightgbm import LGBMRegressor
from pipeline import select_features, create_pipeline
from plots import plot_feature_selection


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
        device="cuda",
        verbose_eval=False,
        verbose=-1,
    )
    estimator_name = "LGBMRegressor"

    best_ablation = None
    previous = (0.156 + 0.120) / 2  # RMSE and MAE Cross target mean value
    current = 0

    i = 0
    candidates = num_cols + cat_cols
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
            save_path="Results/native_feature_selection",
            step_name=f"step_{i + 1}",
            shuffle=False,
            random_state=62,
            imputer="KNNImputer",
            scaler="RobustScaler",
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        i += 1

        # Explicit cleanup: delete temporary variables and force GPU memory free
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass  # If cupy isn't used, ignore

    plot_feature_selection(
        "Results/native_feature_selection", "RMSE", "Results/native_feature_selection"
    )
    plot_feature_selection(
        "Results/native_feature_selection", "MAE", "Results/native_feature_selection"
    )
