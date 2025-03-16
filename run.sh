#!/bin/bash

# Redirect all output (stdout and stderr) to a log file
LOG_FILE="run.log"

# Function to run a command and log it
run_and_log() {
  echo "Running: $1" | tee -a "$LOG_FILE"
  "$1" 2>&1 | tee -a "$LOG_FILE"
  if [ $? -ne 0 ]; then
    echo "Error occurred during: $1" | tee -a "$LOG_FILE"
    exit 1  # Exit script on error
  fi
  echo "Finished: $1" | tee -a "$LOG_FILE"
}

# Create Hold out sets
run_and_log "python datasets.py --methods avg random combinatoric --mode 1"

# Run model selection experiment on Hold out sets
run_and_log "python model_selection.py --run 1 --concat_results 1 --mode ho"

# Generate plots
run_and_log "python analysis_plots.py \"plot_model_selection\""
run_and_log "python analysis_plots.py \"summary_model_selection\""

# Run preprocessing selection
run_and_log "python preprocess_selection.py --run 1 --concat_results 1 --mode ho"

# Generate plots
run_and_log "python analysis_plots.py \"summary_preprocess_selection\""

# Native Feature Selection
run_and_log "python native_feature_selection.py"

# Generate plots
run_and_log "python analysis_plots.py \"plot_native_feature_selection\""

# Feature Engineering and selection
run_and_log "python feature_engineering.py"

# Generate plots
run_and_log "python analysis_plots.py \"plot_feature_engineering\""

# Hyperparameter optimization campaign
run_and_log "python optuna_campaign.py"

# Generate plots
run_and_log "python analysis_plots.py \"plot_optuna_study\""
run_and_log "python analysis_plots.py \"plot_feature_importance_heatmap\""

# Ablation study
run_and_log "python ablation_study.py"

# Generate plots
run_and_log "python analysis_plots.py \"plot_ablation_study\""
run_and_log "python analysis_plots.py \"plot_err_distrib\""
run_and_log "python analysis_plots.py \"plot_err_by_org\""

# Generate other plots
run_and_log "python analysis_plots.py \"plot_global_SHAP\""
run_and_log "python analysis_plots.py \"plot_local_SHAP\""
# run_and_log "python analysis_plots.py \"plot_global_DiCE\""
# run_and_log "python analysis_plots.py \"plot_local_DiCE\""

run_and_log "python analysis_plots.py \"show_perf_skewedness\""

echo "Script execution completed. Check $LOG_FILE for details."
