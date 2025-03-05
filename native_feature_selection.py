import pandas as pd
import numpy as np
import os

from lightgbm import LGBMRegressor
from pipeline import select_features, create_pipeline


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
    estimator = create_pipeline(
        num_cols,
        cat_cols,
        imputer="KNNImputer",
        scaler="RobustScaler",
        estimator=estimator,
        model_name=estimator_name,
    )

    best_ablation = None
    previous = np.inf
    current = 0

    i = 0
    candidates = num_cols + cat_cols
    while previous > current:
        if i != 0:
            # Remove previously eliminated feature
            candidates.pop(best_ablation)
            # Add it to remove_cols
            remove_cols.append(best_ablation)

        current, best_ablation = select_features(
            estimator,
            estimator_name,
            df_dict,
            ho_folder_path="Data/Datasets",
            suffix="_hold_outs.pkl",
            mode="controled_homology",
            target=["Score"],
            candidates=candidates,
            remove_cols=remove_cols,
            save_path="Results/native_feature_selection",
            step_name=f"step_{i + 1}",
            shuffle=False,
            random_state=62,
        )
        i += 1
