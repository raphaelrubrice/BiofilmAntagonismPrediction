#!/bin/bash

# Create Hold out sets and classic 5 fold cv sets
python datasets.py --methods avg random combinatoric --mode cv
python datasets.py --methods avg random combinatoric --mode ho

# Run model selection experiment on Hold out sets
python model_selection.py --run 1 --concat_results 1 --mode ho

# Run preprocessing selection
python preprocess_selection.py --run 1 --concat_results 1 --mode ho

# Generate plots
python plots.py --metrics MAE RMSE R2 --methods avg random combinatoric --plot_model_selection 1 --plot_preprocess_selection 1

