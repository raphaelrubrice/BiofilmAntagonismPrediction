import pandas as pd
import numpy as np
import pickle as pkl
import os
import gc
import cupy as cp

from lightgbm import LGBMRegressor
from pipeline import select_features, create_pipeline, downcast_df
# from plots import plot_feature_selection

if __name__ == "__main__":
    combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_COI.csv")
    combinatoric_df = downcast_df(combinatoric_df)
    # avg_df = pd.read_csv("Data/Datasets/avg_COI.csv")
    df_dict = {"combinatoric": combinatoric_df}
    # df_dict = {"avg": avg_df}

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
        n_jobs=1,
        gpu_use_dp=False,
        tree_learner="serial",
        # device="cuda",
        verbose_eval=False,
        verbose=-1,
    )
    estimator_name = "LGBMRegressor"

    feature_to_remove = None
    previous_metric = None
    cross_mean_metric = None

    i = 0
    candidates = num_cols + cat_cols
    memory = []
    remove_dict = {}
    last_to_remove = []
    # When running parallel evaluation, must pass the path not the actual csv
    # df_dict = {"combinatoric": "Data/Datasets/combinatoric_COI.csv"}
    while len(candidates) > 1:
        if i > 0:
            for feature_to_remove in remove_dict.keys():
                memory.append(feature_to_remove)
                # Remove previously eliminated feature from candidate list and update columns.
                candidates.remove(feature_to_remove)
                if feature_to_remove in num_cols:
                    num_cols.remove(feature_to_remove)
                if feature_to_remove in cat_cols:
                    cat_cols.remove(feature_to_remove)
                remove_cols.append(feature_to_remove)
            previous_metric = (
                cross_mean_metric  # update baseline metric for next iteration
            )

        # Call the feature selection function.
        # It returns: lowest PFI (weighted cross mean), the feature name, and the cross mean.
        last_to_remove = list(remove_dict.keys())
        remove_dict, cross_mean_metric = select_features(
            estimator,
            estimator_name,
            df_dict,
            ho_folder_path="Data/Datasets/",
            suffix="_hold_outs.pkl",
            mode="permutation",
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
            parallel=True,
            n_jobs_outer=12,
            n_jobs_model=1,
            batch_size=12,
        )

        print("*********")
        print(f"Step {i + 1}: Best features to remove and their PFI: {remove_dict}")
        print(f"Last Cross Mean: {previous_metric}")
        print(f"Cross Mean: {cross_mean_metric}")
        print("*********")

        # Stop if the error (PFI/cross mean) increased compared to the previous iteration.
        if i == 0:
            previous_metric = cross_mean_metric
        if cross_mean_metric > previous_metric:
            # Baseline will be the error when rmeoving last feature
            # so if error was increased, removing the last feature was a bad idea
            for feature in last_to_remove:
                memory.remove(feature)
            break

        i += 1

        # Cleanup: garbage collection and freeing GPU memory.
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    with open("./Results/native_feature_selection/to_remove.pkl", "wb") as f:
        pkl.dump(memory, f)
    # Optionally, call the plotting function.
    # plot_feature_selection("Results/native_feature_selection", "Results/native_feature_selection")
