import fireducks.pandas as pd
import pickle as pkl
import os
import gc
import cupy as cp

from pipeline import make_best_estimator, evaluate
from lightgbm import LGBMRegressor

if __name__ == '__main__':
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
    os.makedirs("./Results/reco_exp/impute_bias/", exist_ok=True)

    for stratify in [False, True]:
        if stratify:
            for mode in ['Quantile', 'Custom']:
                for mixed in [True, False]:
                    strat_params = {'mode':mode.lower(), 
                                    'ranges':[0.2, 0.4, 0.6, 0.8],
                                    'mixed_training':mixed,
                                    'random_state':6262}
                    prefix = 'Stratified'
                    mixed_addon = 'Mixed' if mixed else 'Default'

                    for nan_flag in ['NoImpute', 'Impute']:
                        model_class = LGBMRegressor
                        model_name = prefix + 'LGBMRegressor'

                        if nan_flag == 'NoImpute':
                            estimator = make_best_estimator(model_class, 
                                                            model_name,
                                                            stratified=stratify, 
                                                            stratify_params=strat_params, 
                                                            no_imputation=True)
                        else:
                            estimator = make_best_estimator(model_class,
                                                            model_name,
                                                            stratified=stratify, 
                                                            stratify_params=strat_params)

                        
                        # 2 * 6 => 12 threads under the hood
                        outer_threads = 2

                        save_models_path = f"./Results/reco_exp_models/impute_bias/"
                        os.makedirs(save_models_path, exist_ok=True)
                        full_name = f"{nan_flag}_{mode}_{mixed_addon}_{model_name}"
                        results = evaluate(
                                estimator,
                                full_name + '_',
                                df_dict,
                                mode="ho",
                                suffix="_hold_outs.pkl",
                                ho_folder_path="Data/Datasets/",
                                target=target,
                                remove_cols=remove_cols,
                                save=True,
                                save_path=save_models_path,
                                parallel=True,
                                n_jobs_outer=outer_threads,
                                n_jobs_model=1,
                                batch_size=outer_threads,
                                temp_folder="./temp_results",
                            )
                        results.to_csv(f"Results/reco_exp/impute_bias/ho_{full_name}_results.csv")

                        del results
                        gc.collect()
                        cp.get_default_memory_pool().free_all_blocks()
        else:
            strat_params = {}
            prefix = 'Normal'

            for nan_flag in ['NoImpute', 'Impute']:
                model_class = LGBMRegressor
                model_name = prefix + 'LGBMRegressor'

                if nan_flag == 'NoImpute':
                    estimator = make_best_estimator(model_class, 
                                                    model_name,
                                                    stratified=stratify, 
                                                    stratify_params=strat_params, 
                                                    no_imputation=True)
                else:
                    estimator = make_best_estimator(model_class,
                                                    model_name,
                                                    stratified=stratify, 
                                                    stratify_params=strat_params)

            
                outer_threads = 12

                save_models_path = f"./Results/reco_exp_models/impute_bias/"
                os.makedirs(save_models_path, exist_ok=True)
                full_name = f"{nan_flag}_{model_name}"
                results = evaluate(
                        estimator,
                        full_name + '_',
                        df_dict,
                        mode="ho",
                        suffix="_hold_outs.pkl",
                        ho_folder_path="Data/Datasets/",
                        target=target,
                        remove_cols=remove_cols,
                        save=True,
                        save_path=save_models_path,
                        parallel=True,
                        n_jobs_outer=outer_threads,
                        n_jobs_model=1,
                        batch_size=outer_threads,
                        temp_folder="./temp_results",
                    )
                results.to_csv(f"Results/reco_exp/impute_bias/ho_{full_name}_results.csv")

                del results
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
    del combinatoric_df, df_dict