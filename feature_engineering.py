import pandas as pd
import numpy as np
import pickle as pkl
import os
import gc
import cupy as cp

from pipeline import create_pipeline, select_features
from datasets import make_feature_engineered_dataset
# from plots import plot_feature_selection

from lightgbm import LGBMRegressor


if __name__ == "__main__":
    combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_COI.csv")
    # avg_df = pd.read_csv("Data/Datasets/avg_COI.csv")
    df_dict = {"combinatoric": combinatoric_df}
    # df_dict = {"avg": avg_df}

    target = ["Score"]
    cat_cols = ["Modele"]
    # Retrieve native features that were excluded
    with open("./Results/native_feature_selection/to_remove.pkl", "rb") as f:
        to_remove = pkl.load(f)

    remove_cols = [
        "Unnamed: 0",
        "B_sample_ID",
        "P_sample_ID",
        "Bacillus",
        "Pathogene",
    ] + to_remove

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

    i = 0
    candidates = num_cols + cat_cols

    # Save Feature Engineering to avoid recomputation at each step
    FE_combinatoric_df = make_feature_engineered_dataset(
        combinatoric_df,
        "Data/Datasets/fe_combinatoric_COI.csv",
        cols_prod=num_cols,
        cols_diff=num_cols,
        cols_pow=num_cols,
        pow_orders=[2, 3],
        target=target,
        remove_cols=remove_cols,
    )
    # FE_avg_df = make_feature_engineered_dataset(
    #     avg_df,
    #     "Data/Datasets/avg_FeatureEng.csv",
    #     cols_prod=candidates,
    #     cols_diff=num_cols,
    #     cols_pow=num_cols,
    #     pow_orders=[2, 3],
    #     target=target,
    #     remove_cols=remove_cols,
    # )

    df_dict = {"combinatoric": FE_combinatoric_df}
    # df_dict = {"avg": FE_avg_df}
    num_cols += [
        col
        for col in FE_combinatoric_df.columns
        if col not in num_cols + cat_cols + remove_cols + target
    ]
    feature_to_remove = None
    previous_metric = None
    cross_mean_metric = None

    i = 0
    candidates = num_cols + cat_cols
    memory = []
    remove_dict = {}
    last_to_remove = []
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
            save_path="Results/feature_engineering",
            step_name=f"step_{i + 1}",
            shuffle=False,
            random_state=62,
            imputer="KNNImputer",
            scaler="RobustScaler",
            num_cols=num_cols,
            cat_cols=cat_cols,
            parallel=True,
            n_jobs_outer=8,
            n_jobs_model=1,
            batch_size=8,
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
            with open("./Results/feature_engineering/best_score.pkl", "wb") as f:
                pkl.dump(previous_metric, f)
            break

        i += 1

        # Cleanup: garbage collection and freeing GPU memory.
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    with open("./Results/feature_engineering/to_remove.pkl", "wb") as f:
        pkl.dump(memory, f)

    FE_combinatoric_df[
        [col for col in FE_combinatoric_df.columns if col not in memory + to_remove]
    ].to_csv("./Data/Datasets/fe_combinatoric_COI.csv")
