#!/bin/bash

# Generate plots
python analysis_plots.py "plot_model_selection"
python analysis_plots.py "summary_model_selection"


python analysis_plots.py "summary_preprocess_selection"


python analysis_plots.py "plot_native_feature_selection"

python analysis_plots.py "plot_feature_engineering"

python analysis_plots.py "plot_optuna_study"
python analysis_plots.py "plot_feature_importance_heatmap"

python analysis_plots.py "plot_ablation_study"
python analysis_plots.py "plot_err_distrib"
python analysis_plots.py "plot_err_by_org"

python analysis_plots.py "plot_global_SHAP"
python analysis_plots.py "plot_local_SHAP"
python analysis_plots.py "plot_global_DiCE"
python analysis_plots.py "plot_local_DiCE"

python analysis_plots.py "show_perf_skewedness"