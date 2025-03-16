#!/bin/bash

# Create Hold out sets
echo "Starting: python datasets.py --methods avg random combinatoric --mode 1"
python datasets.py --methods avg random combinatoric --mode 1
echo "Finished: python datasets.py --methods avg random combinatoric --mode 1"

# Run model selection experiment on Hold out sets
echo "Starting: python model_selection.py --run 1 --concat_results 1 --mode ho"
python model_selection.py --run 1 --concat_results 1 --mode ho
echo "Finished: python model_selection.py --run 1 --concat_results 1 --mode ho"

# Generate plots
echo "Starting: python analysis_plots.py \"plot_model_selection\""
python analysis_plots.py "plot_model_selection"
echo "Finished: python analysis_plots.py \"plot_model_selection\""

echo "Starting: python analysis_plots.py \"summary_model_selection\""
python analysis_plots.py "summary_model_selection"
echo "Finished: python analysis_plots.py \"summary_model_selection\""

# Run preprocessing selection
echo "Starting: python preprocess_selection.py --run 1 --concat_results 1 --mode ho"
python preprocess_selection.py --run 1 --concat_results 1 --mode ho
echo "Finished: python preprocess_selection.py --run 1 --concat_results 1 --mode ho"

# Generate plots
echo "Starting: python analysis_plots.py \"summary_preprocess_selection\""
python analysis_plots.py "summary_preprocess_selection"
echo "Finished: python analysis_plots.py \"summary_preprocess_selection\""

# Native Feature Selection
echo "Starting: python native_feature_selection.py"
python native_feature_selection.py
echo "Finished: python native_feature_selection.py"

# Generate plots
echo "Starting: python analysis_plots.py \"plot_native_feature_selection\""
python analysis_plots.py "plot_native_feature_selection"
echo "Finished: python analysis_plots.py \"plot_native_feature_selection\""

# Feature Engineering and selection
echo "Starting: python feature_engineering.py"
python feature_engineering.py
echo "Finished: python feature_engineering.py"

# Generate plots
echo "Starting: python analysis_plots.py \"plot_feature_engineering\""
python analysis_plots.py "plot_feature_engineering"
echo "Finished: python analysis_plots.py \"plot_feature_engineering\""

# Hyperparameter optimization campaign
echo "Starting: python optuna_campaign.py"
python optuna_campaign.py
echo "Finished: python optuna_campaign.py"

# Generate plots
echo "Starting: python analysis_plots.py \"plot_optuna_study\""
python analysis_plots.py "plot_optuna_study"
echo "Finished: python analysis_plots.py \"plot_optuna_study\""

echo "Starting: python analysis_plots.py \"plot_feature_importance_heatmap\""
python analysis_plots.py "plot_feature_importance_heatmap"
echo "Finished: python analysis_plots.py \"plot_feature_importance_heatmap\""

# Ablation study
echo "Starting: python ablation_study.py"
python ablation_study.py
echo "Finished: python ablation_study.py"

# Generate plots
echo "Starting: python analysis_plots.py \"plot_ablation_study\""
python analysis_plots.py "plot_ablation_study"
echo "Finished: python analysis_plots.py \"plot_ablation_study\""

echo "Starting: python analysis_plots.py \"plot_err_distrib\""
python analysis_plots.py "plot_err_distrib"
echo "Finished: python analysis_plots.py \"plot_err_distrib\""

echo "Starting: python analysis_plots.py \"plot_err_by_org\""
python analysis_plots.py "plot_err_by_org"
echo "Finished: python analysis_plots.py \"plot_err_by_org\""

# Generate other plots
echo "Starting: python analysis_plots.py \"plot_global_SHAP\""
python analysis_plots.py "plot_global_SHAP"
echo "Finished: python analysis_plots.py \"plot_global_SHAP\""

echo "Starting: python analysis_plots.py \"plot_local_SHAP\""
python analysis_plots.py "plot_local_SHAP"
echo "Finished: python analysis_plots.py \"plot_local_SHAP\""

echo "Starting: python analysis_plots.py \"plot_global_DiCE\""
python analysis_plots.py "plot_global_DiCE"
echo "Finished: python analysis_plots.py \"plot_global_DiCE\""

echo "Starting: python analysis_plots.py \"plot_local_DiCE\""
python analysis_plots.py "plot_local_DiCE"
echo "Finished: python analysis_plots.py \"plot_local_DiCE\""

echo "Starting: python analysis_plots.py \"show_perf_skewedness\""
python analysis_plots.py "show_perf_skewedness"
echo "Finished: python analysis_plots.py \"show_perf_skewedness\""

echo "Script execution completed."