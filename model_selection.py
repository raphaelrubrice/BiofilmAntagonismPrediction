import pandas as pd
import os
import argparse

from pipeline import create_pipeline, evaluate
from utils import is_gpu_available

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


gpu_available = is_gpu_available()

if gpu_available:
    # Use cuML’s versions where available
    from cuml.linear_model import (
        LinearRegression as cuMLLinearRegression,
        Ridge as cuMLRidge,
    )
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
else:
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
    )
    from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs="?",
        default=0,
        type=int,
        help="Run whole experiment. 0 by deault",
    )
    parser.add_argument(
        "--concat_results",
        nargs="?",
        default=0,
        type=int,
        help="Concatenate all results (1) or not (0=default)",
    )
    parser.add_argument(
        "--mode",
        nargs="?",
        default="ho",
        type=str,
        help="Run experiment using homology controled sets or classic cv (pass cv). Default is 'ho'.",
    )

    args = parser.parse_args()

    if args.run:
        METHOD = "combinatoric"
        MODE = args.mode
        PROTOCOL_SUFFIX = "_hold_outs.pkl" if MODE != "cv" else "_cv.pkl"

        model_dict = (
            {
                "LinearRegression": cuMLLinearRegression(),
                "Ridge": cuMLRidge(),
                "Lasso": Lasso(),  # fallback (no GPU version available)
                "ElasticNet": ElasticNet(),  # fallback
                "LinearSVR": LinearSVR(),  # fallback
                "RandomForestRegressor": cuMLRandomForestRegressor(random_state=62),
                "GradientBoostingRegressor": GradientBoostingRegressor(
                    random_state=62
                ),  # fallback
                # Use GPU-accelerated LightGBM
                "LGBMRegressor": LGBMRegressor(
                    random_state=62,
                    n_jobs=-1,
                    verbose_eval=False,
                    verbose=-1,
                    device="cuda",
                ),
                # Use XGBRegressor with GPU options
                "XGBRegressor": XGBRegressor(
                    n_estimators=100,
                    random_state=62,
                    n_jobs=-1,
                    tree_method="gpu_hist",
                    predictor="gpu_predictor",
                    gpu_id=0,
                ),
            }
            if gpu_available
            else {
                "LinearRegression": LinearRegression(n_jobs=-1),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "LinearSVR": LinearSVR(),
                "RandomForestRegressor": RandomForestRegressor(
                    random_state=62, n_jobs=-1
                ),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=62),
                "LGBMRegressor": LGBMRegressor(
                    random_state=62, n_jobs=-1, verbose_eval=False, verbose=-1
                ),
                "XGBRegressor": XGBRegressor(
                    n_estimators=100, random_state=62, n_jobs=-1
                ),
            }
        )

        avg_df = pd.read_csv("Data/Datasets/avg_COI.csv")
        random_df = pd.read_csv("Data/Datasets/random_COI.csv")
        combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_COI.csv")

        df_dict = {"avg": avg_df, "random": random_df, "combinatoric": combinatoric_df}

        target = ["Score"]
        cat_cols = ["Modele"]
        remove_cols = (
            ["Unnamed: 0", "B_sample_ID", "P_sample_ID", "Bacillus", "Pathogene"]
            if METHOD == "combinatoric"
            else ["Unnamed: 0", "Bacillus", "Pathogene"]
        )
        num_cols = [
            col
            for col in df_dict["combinatoric"].columns
            if col not in cat_cols + remove_cols + target
        ]

        os.makedirs("Results/model_selection/", exist_ok=True)
        for model_name, estimator in model_dict.items():
            estimator = create_pipeline(
                num_cols,
                cat_cols,
                imputer="MeanImputer",
                scaler="StandardScaler",
                estimator=estimator,
                model_name=model_name,
            )
            results = evaluate(
                estimator,
                model_name,
                df_dict,
                mode=MODE,
                suffix=PROTOCOL_SUFFIX,
                ho_folder_path="Data/Datasets/",
                target=target,
                remove_cols=remove_cols,
            )

            results.to_csv(f"Results/model_selection/{MODE}_{model_name}_results.csv")
    if args.concat_results:
        all_results = []
        for file in os.listdir("Results/model_selection"):
            if file.endswith(".csv") and mode in file and "_all_results" not in file:
                all_results.append(pd.read_csv("Results/model_selection/" + file))
        if all_results != []:
            all_df = pd.concat(all_results, axis=0)
            all_df.to_csv(f"Results/model_selection/{mode}_all_results.csv")
        else:
            print("No result files found.")
