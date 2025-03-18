from analysis_plots import *

plot_model_selection(
    "./Results/model_selection/ho_all_results.csv",
    "MAE",
    mode="all",
    avg_mode="weighted",
    save_path="./Plots/local/model_selection_MAE.pdf",
    show=False,
)

plot_model_selection(
    "./Results/model_selection/ho_all_results.csv",
    "RMSE",
    mode="all",
    avg_mode="weighted",
    save_path="./Plots/local/model_selection_RMSE.pdf",
    show=False,
)

summary_model_selection(
    "./Results/model_selection/ho_all_results.csv",
    metric="MAE",
    method="combinatoric",
    avg_mode="weighted",
    save_path="./Plots/local/summary_model_selection_MAE.pdf",
    show=False,
)

summary_model_selection(
    "./Results/model_selection/ho_all_results.csv",
    metric="RMSE",
    method="combinatoric",
    avg_mode="weighted",
    save_path="./Plots/local/summary_model_selection_RMSE.pdf",
    show=False,
)

summary_preprocess_selection(
    "./Results/preprocess_selection/ho_all_results.csv",
    metric="MAE",
    method="combinatoric",
    avg_mode="weighted",
    save_path="./Plots/local/preprocess_selection_MAE.pdf",
    show=False,
)


summary_preprocess_selection(
    "./Results/preprocess_selection/ho_all_results.csv",
    metric="RMSE",
    method="combinatoric",
    avg_mode="weighted",
    save_path="./Plots/local/preprocess_selection_RMSE.pdf",
    show=False,
)


plot_native_feature_selection(
    "./Results/native_feature_selection/step_1_LGBMRegressor_controled_homology_permutation_details.csv",
    ci_mode="bca",
    save_path="./Plots/local/native_feature_selection.pdf",
    show=False,
)


plot_feature_engineering(
    "./Results/feature_engineering/step_1_LGBMRegressor_controled_homology_permutation_details.csv",
    ci_mode="bca",
    save_path="./Plots/local/feature_engineering",
    show=False,
)


plot_optuna_study(
    "./Results/optuna_campaign/optuna_study.pkl",
    save_path="./Plots/local/optuna_study",
    show=False,
)

plot_feature_importance_heatmap(
    "./Results/models/",
    save_path="./Plots/local/feature_importances_GAIN.pdf",
    show=False,
)

plot_ablation_study(
    "./Results/ablation_study/",
    save_path="./Plots/local/ablation_study",
    show=False,
)

plot_err_distrib(
    "./Results/ablation_study/ho_None_LGBMRegressor_results.csv",
    save_path="./Plots/local/distrib_err",
    show=False,
)

plot_err_by_org(
    "./Results/ablation_study/ho_None_LGBMRegressor_results.csv",
    save_path="./Plots/local/err_by_org",
    show=False,
)

plot_global_SHAP(
    ho_name="1234_x_S.en", save_path="./Plots/local/global_SHAP", show=False
)


plot_global_SHAP(
    ho_name="12001_x_E.ce", save_path="./Plots/local/global_SHAP", show=False
)

plot_local_SHAP(
    ho_name="1234_x_S.en",
    mode="worst",
    save_path="./Plots/local/local_SHAP",
    show=False,
)

plot_local_SHAP(
    ho_name="1234_x_S.en",
    mode="best",
    save_path="./Plots/local/local_SHAP",
    show=False,
)
