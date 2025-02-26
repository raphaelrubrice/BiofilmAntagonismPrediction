import pandas as pd
import json
import os
import argparse

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from pipeline import create_pipeline, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default=0, type=int)
    parser.add_argument("concat_results", nargs="?", default=0, type=int)
    parser.add_argument("mode", nargs="?", default="ho", type=str)

    args = parser.parse_args()

    if args.run:
        METHOD = "combinatoric"
        MODE = args.mode
        PROTOCOL_SUFFIX = "_hold_outs.pkl" if MODE != "cv" else "_cv.pkl"

        model_dict = {
            "LinearRegression": LinearRegression(n_jobs=-1),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "LinearSVR": LinearSVR(),
            # "SVR": SVR(kernel='rbf'),
            "RandomForestRegressor": RandomForestRegressor(random_state=62, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=62),
            "LGBMRegressor": LGBMRegressor(random_state=62, n_jobs=-1, verbose_eval=False, verbose=-1),
            "XGBRegressor":XGBRegressor(n_estimators=100, random_state=62, n_jobs=-1)
        }

        avg_df = pd.read_csv("Data/Datasets/avg_COI.csv")
        random_df = pd.read_csv("Data/Datasets/random_COI.csv")
        combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_COI.csv")

        df_dict = {
                    "avg":avg_df,
                "random":random_df,
                "combinatoric":combinatoric_df
                }
        
        target = ["Score"]
        cat_cols = ["Modele"]
        remove_cols = ['Unnamed: 0', "B_sample_ID", "P_sample_ID", "Bacillus", "Pathogene"] if METHOD == "combinatoric" else ['Unnamed: 0', "Bacillus", "Pathogene"]
        num_cols = [col for col in df_dict["combinatoric"].columns if col not in cat_cols + remove_cols + target]

        os.makedirs(f"Results/model_selection/", exist_ok=True)
        for model_name, estimator in model_dict.items():

            estimator = create_pipeline(num_cols, cat_cols, imputer="MeanImputer", scaler="StandardScaler", 
                                        estimator=estimator, model_name=model_name)
            results = evaluate(estimator, model_name, df_dict, mode=MODE, suffix=PROTOCOL_SUFFIX,
                    ho_folder_path="Data/Datasets/", target=target, 
                    remove_cols=remove_cols)
            
            results.to_csv(f"Results/model_selection/{MODE}_{model_name}_results.csv")
    if args.concat_results:
        all_results = []
        for file in os.listdir("Results/model_selection"):
            if file.endswith(".csv"):
                all_results.append(pd.read_csv("Results/model_selection/" + file))
        if all_results != []:
            all_df = pd.concat(all_results, axis=0)
            all_df.to_csv("Results/model_selection/ho_all_results.csv")
        else:
            print("No result files found.")