import pandas as pd
import os
import argparse

from lightgbm import LGBMRegressor
from pipeline import create_pipeline, evaluate

# During model and dataset selection we identified the LGBMRegressor
# and the combinaoric dataset as the best ones.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", nargs="?", default=0, type=int, help="Run the evaluation."
    )
    parser.add_argument(
        "--concat_results",
        nargs="?",
        default=0,
        type=int,
        help="Concatenate result files.",
    )
    parser.add_argument(
        "--mode",
        nargs="?",
        default="ho",
        type=str,
        help="Evaluation mode: 'ho' or 'cv'.",
    )

    args = parser.parse_args()

    if args.run == 1:
        # Define evaluation parameters
        METHOD = "combinatoric"
        MODE = args.mode
        PROTOCOL_SUFFIX = "_hold_outs.pkl" if MODE != "cv" else "_cv.pkl"

        # Load datasets
        avg_df = pd.read_csv("Data/Datasets/avg_COI.csv")
        random_df = pd.read_csv("Data/Datasets/random_COI.csv")
        combinatoric_df = pd.read_csv("Data/Datasets/combinatoric_COI.csv")

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
            for col in combinatoric_df.columns
            if col not in cat_cols + remove_cols + target
        ]

        os.makedirs("Results/preprocess_selection/", exist_ok=True)

        # Define imputer and scaler options
        imputer_options = [
            "MeanImputer",
            "MedianImputer",
            "KNNImputer",
            "RandomForestImputer",
        ]
        scaler_options = [
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "MaxAbsScaler",
        ]

        # Build pipeline dictionary for LGBMRegressor
        pipeline_dict = {}
        for imputer in imputer_options:
            for scaler in scaler_options:
                key = f"{imputer}_{scaler}"
                pipeline = create_pipeline(
                    num_cols,
                    cat_cols,
                    imputer=imputer,
                    scaler=scaler,
                    estimator=LGBMRegressor(
                        random_state=62,
                        n_jobs=-1,
                        gpu_use_dp=False,
                        max_bin=63,
                        tree_learner="serial",
                        device="cuda",
                        verbose_eval=False,
                        verbose=-1,
                    ),
                    model_name="LGBMRegressor",
                )
                pipeline_dict[key] = pipeline

        # Evaluate each pipeline and save the results
        for key, pipeline in pipeline_dict.items():
            results = evaluate(
                pipeline,
                key,
                df_dict,
                mode=MODE,
                suffix=PROTOCOL_SUFFIX,
                ho_folder_path="Data/Datasets/",
                target=target,
                remove_cols=remove_cols,
            )
            # If results is not already a DataFrame, convert it.
            results_df = (
                results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
            )
            results_df.to_csv(
                f"Results/preprocess_selection/{key}_LGBMRegressor_{MODE}_results.csv",
                index=False,
            )

    if args.concat_results:
        all_results = []
        for file in os.listdir("Results/model_selection"):
            if (
                file.endswith(".csv")
                and args.mode in file
                and "_all_results" not in file
            ):
                all_results.append(pd.read_csv("Results/model_selection/" + file))
        if all_results != []:
            all_df = pd.concat(all_results, axis=0)
            all_df.to_csv(f"Results/model_selection/{args.mode}_all_results.csv")
        else:
            print("No result files found.")

    if args.concat_results:
        # Concatenate all CSV result files into one file.
        all_results = []
        results_folder = "Results/preprocess_selection"
        for file in os.listdir(results_folder):
            if (
                file.endswith(".csv")
                and args.mode in file
                and "_all_results" not in file
            ):
                all_results.append(pd.read_csv(os.path.join(results_folder, file)))
        if all_results:
            all_df = pd.concat(all_results, axis=0)
            all_df.to_csv(
                os.path.join(results_folder, f"{args.mode}_all_results.csv"),
                index=False,
            )
        else:
            print("No result files found.")
