import pandas as pd
import numpy as np
import pickle as pkl
import os, re, argparse, glob, traceback

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.lines import Line2D

import lightgbm as lgb
import optuna
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_param_importances,
    plot_rank,
    plot_parallel_coordinate,
)
import shap
import dice_ml

from scipy.stats import norm, t
from statsmodels.graphics.gofplots import qqplot
from arch.bootstrap import IIDBootstrap

from datasets import all_possible_hold_outs, get_hold_out_sets, get_train_test_split

all_ho_names = all_possible_hold_outs(return_names=True)
ho_org = [name for name in all_ho_names if "_x_" not in name]
ho_interaction = [name for name in all_ho_names if name not in ho_org]
ho_pathogen = ["E.ce", "E.co", "S.au", "S.en"]
ho_bacillus = [name for name in ho_org if name not in ho_pathogen]

sns.set_theme()


def show_perf_skewedness(
    model_name, method="combinatoric", path_df=None, save_path=None, show=False
):
    """
    Display the distribution of model performance metrics (MAE and RMSE) across different task types.
    The function plots kernel density estimates (KDE) of error distributions, highlighting the mean values.

    Parameters:
    - model_name (str): Name of the model to filter results.
    - method (str): Not currently used, but reserved for future extensions.
    - path_df (str, optional): Path to CSV file containing evaluation results. Defaults to None.
    """

    if path_df is None:
        results = pd.read_csv("./Results/model_selection/ho_all_results.csv")
    else:
        results = pd.read_csv(path_df)

    # Filter results for the given model
    plot_df = results[results["Model"] == model_name]

    # Categorize evaluation tasks
    evaltype = []
    for row in range(plot_df.shape[0]):
        eval_label = plot_df["Evaluation"].iloc[row]
        if eval_label in ["E.ce", "E.co", "S.au", "S.en"]:
            evaltype.append("Pathogen")
        elif "_x_" in eval_label:
            evaltype.append("Interaction")
        else:
            evaltype.append("Bacillus")

    plot_df["Task"] = evaltype

    # Set seaborn style for a professional look
    sns.set_theme(style="whitegrid")

    # Define figure size for readability
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # KDE Plot for MAE
    sns.kdeplot(data=plot_df, x="MAE", ax=axes[0], hue="Task", fill=True, alpha=0.6)
    sns.kdeplot(plot_df, x="MAE", ax=axes[0], color="orange")
    axes[0].axvline(np.mean(plot_df["MAE"]), linestyle="--", color="red")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Mean Absolute Error (MAE)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Density", fontsize=14, fontweight="bold")
    axes[0].set_title("MAE Distribution Across Tasks", fontsize=16, fontweight="bold")
    # axes[0].legend(fontsize=12)

    # KDE Plot for RMSE
    sns.kdeplot(data=plot_df, x="RMSE", ax=axes[1], hue="Task", fill=True, alpha=0.6)
    sns.kdeplot(plot_df, x="RMSE", ax=axes[1], color="orange")
    axes[1].axvline(
        np.mean(plot_df["RMSE"]), linestyle="--", color="red", label="Mean RMSE"
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Root Mean Squared Error (RMSE)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Density", fontsize=14, fontweight="bold")
    axes[1].set_title("RMSE Distribution Across Tasks", fontsize=16, fontweight="bold")
    axes[1].legend(fontsize=12)

    # Improve spacing
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


# Because we have heterogeneous folds (18 Bacillus, 4 Pathogen, 72 Interactions)
# and because model performances are themselves skewed (see skewedness plots)
# we found ourselves in a setup where normality assumptions regarding performances may not hold.
# Thus we will be computing confidence intervals around the mean using Bias Corrected acceleration,
# (bootstrap) which is more appropriate for computing skewed and biased distributions and
# stays appropriate for normally distributed data.
# Concerns about BCa properties invalidity due to small sample size do not apply
# as we have 94 folds which is superior to the 30 threshold usually used as minimum for validity.
def stat_func(data):
    return np.mean(data)


def weighted_stat_func(data):
    return np.sum(data)


def compute_CI(
    data, num_iter=5000, confidence=95, seed=None, stat_func=stat_func, mode="bca"
):
    if confidence > 1:
        confidence = confidence / 100
    if mode == "bca":
        bs = IIDBootstrap(data, seed=seed)
        low, up = bs.conf_int(stat_func, reps=num_iter, method="bca", size=confidence)
        return abs(low[0]), abs(up[0])
    else:
        avg = np.mean(data)
        std = np.std(data)
        low, up = t.interval(confidence, data.shape[0] - 1, loc=avg, scale=std)
        return abs(low), abs(up)


# Mettre en annexe le qq plot student pour dire que bien que loin
# d'être parfait, l'alignement est correct. On peut donc faire
# l'hypothese simplificatrice que notre distribution des scores
# suit une loi de student, et donc utiliser cela pour calculer les
# intervalles de confiance en utilsant le coefficient de student.
def check_qqplot(data, dist, save_path=None, show=False):
    avg = np.mean(data)
    std = np.std(data)
    dist_dico = {"t": t, "norm": norm}
    distargs = () if dist != "t" else (data.shape[0] - 1,)
    dist = dist_dico[dist]
    qqplot(data, dist=dist, distargs=distargs, loc=avg, scale=std)

    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def plot_model_selection(
    results_file_path,
    metric="RMSE",
    mode="all",
    avg_mode="unweighted",
    save_path=None,
    show=False,
):
    """
    Plot model selection performance with 95% confidence intervals.

    Parameters:
    - results_file_path (str): Path to CSV file containing model selection results.
    - metric (str): Metric to plot (RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2).
    - mode (str): Subset of data to plot ("bacillus", "pathogen", "interaction", "all").
    - avg_mode (str): Type of average to compute ("weighted", "unweighted").
    - save_path (str, optional): Path to save the plot.
    - show (bool): Whether to display the plot.
    """

    # Validate metric choice
    assert metric in [
        "RMSE",
        "MAE",
        "std_abs_err",
        "MAPE",
        "std_abs_relative_err",
        "R2",
    ], "metric must be one of: RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2"

    # Validate avg_mode choice
    assert avg_mode in ["weighted", "unweighted"], (
        "avg_mode must be 'weighted' or 'unweighted'"
    )

    # Read results from CSV
    results = pd.read_csv(results_file_path)

    # Filter results based on mode
    if mode == "bacillus":
        results = results[results["Evaluation"].isin(ho_bacillus)]
    elif mode == "pathogen":
        results = results[results["Evaluation"].isin(ho_pathogen)]
    elif mode == "interaction":
        results = results[results["Evaluation"].isin(ho_interaction)]
    else:
        results = results[
            results["Evaluation"].isin(ho_bacillus + ho_pathogen + ho_interaction)
        ]

    # Define aggregation function based on avg_mode
    if avg_mode == "weighted":
        agg_data = (
            results.groupby(["Method", "Model"])
            .apply(
                lambda g: np.sum(g[metric] * g["n_samples"]) / np.sum(g["n_samples"])
            )
            .reset_index(name="mean_metric")
        )
    else:
        agg_data = (
            results.groupby(["Method", "Model"])
            .agg(mean_metric=(metric, "mean"))
            .reset_index()
        )

    # Compute confidence intervals using compute_CI function
    if avg_mode == "weighted":
        ci_data = (
            results.groupby(["Method", "Model"])
            .apply(
                lambda g: pd.Series(
                    compute_CI(
                        g[metric] * g["n_samples"] / np.sum(g["n_samples"]),
                        mode="bca",
                        stat_func=weighted_stat_func,
                    )
                )
            )
            .rename(columns={0: "ci_low", 1: "ci_up"})
            .reset_index()
        )
    else:
        ci_data = (
            results.groupby(["Method", "Model"])
            .apply(lambda g: pd.Series(compute_CI(g[metric].values, mode="bca")))
            .rename(columns={0: "ci_low", 1: "ci_up"})
            .reset_index()
        )

    # Merge dataframes
    agg_data = pd.merge(agg_data, ci_data, on=["Method", "Model"])

    # Calculate CI half-width for error bars
    agg_data["ci"] = abs(agg_data["mean_metric"] - agg_data["ci_low"])

    # Sort data by the mean value for plotting
    agg_data = agg_data.sort_values("mean_metric")
    y_value = "mean_metric"
    error_column = "ci"

    # Compute overall best (lowest) value of the metric
    best_val = agg_data[y_value].min() if metric != "R2" else agg_data[y_value].max()
    best_model = agg_data["Model"][agg_data[y_value] == best_val].iloc[0]
    best_method = agg_data["Method"][agg_data[y_value] == best_val].iloc[0]

    # Set publication-grade style settings
    sns.set_theme(style="whitegrid", font="serif", font_scale=1)
    plt.rcParams.update(
        {"font.size": 8, "font.family": "sans-serif", "font.weight": "bold"}
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))

    # Draw barplot with seaborn
    ax = sns.barplot(
        data=agg_data,
        x="Model",
        y=y_value,
        hue="Method",
        palette="inferno",
        ax=ax,
        errorbar=None,
        hue_order=["random", "avg", "combinatoric"],
        order=[
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "LinearSVR",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "LGBMRegressor",
            "XGBRegressor",
        ],
    )

    # Add error bars using the computed 95% CI
    for patch, ci in zip(ax.patches, agg_data[error_column].values):
        x_center = patch.get_x() + patch.get_width() / 2.0
        height = patch.get_height()
        ax.errorbar(x_center, height, yerr=ci, color="black", capsize=5, fmt="none")

    # Add a horizontal dotted line at the overall best value and label it
    ax.axhline(
        y=best_val,
        linestyle=":",
        color="red",
        linewidth=2,
        label=f"overall best: {best_val:.3f}",
    )

    # Add title and axis labels with improved formatting
    avg_type = "Weighted" if avg_mode == "weighted" else "Unweighted"
    ax.set_title(
        f"Model Selection Performance ({metric}) - {avg_type} Average\n Best Model: {best_model} | Best Method: {best_method}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric, fontsize=14, fontweight="bold")

    # Create a custom legend entry for the overall best line
    handles, labels = ax.get_legend_handles_labels()
    best_handle = Line2D(
        [],
        [],
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"overall best: {best_val:.3f}",
    )
    if f"overall best: {best_val:.3f}" not in labels:
        handles.append(best_handle)
        labels.append(f"overall best: {best_val:.3f}")

    ax.legend(
        handles=handles, labels=labels, title="Method", title_fontsize=12, fontsize=10
    )

    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path is not None:
        if save_path.endswith(".pdf"):
            save_path = (
                save_path[: save_path.index(".pdf")]
                + f"_{mode}_{avg_mode}"
                + save_path[save_path.index(".pdf") :]
            )
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path + f"_{mode}_{avg_mode}" + ".pdf",
                format="pdf",
                bbox_inches="tight",
            )
    if show:
        plt.show()


def summary_model_selection(
    results_file_path,
    metric="RMSE",
    method="avg",
    avg_mode="unweighted",
    save_path=None,
    show=False,
):
    """
    Produces a summary plot showing, for each model, four bars corresponding to:
        - Overall: Mean computed on all evaluation sets (ho_bacillus + ho_pathogen + ho_interaction)
        - Bacillus: Mean computed on evaluation sets in ho_bacillus
        - Pathogen: Mean computed on evaluation sets in ho_pathogen
        - Interaction: Mean computed on evaluation sets in ho_interaction

    Each bar displays the average performance (using the specified metric) and its 95% confidence interval.
    The plot also shows a horizontal dotted line at the best (lowest) Overall mean, with a legend entry.

    Parameters:
        results_file_path : str
            Path to the CSV file containing the results.
        metric : str
            One of ["RMSE", "MAE", "std_abs_err", "MAPE", "std_abs_relative_err", "R2"].
        method : str
            The method to filter by (e.g. "avg", "random", or "combinatoric").
        avg_mode : str
            Averaging mode, either "weighted" (weighted by n_samples) or "unweighted" (default).
        save_path : str, optional
            If provided, the plot is saved as a PDF.
        show : bool, optional
            If True, the plot is displayed.

    Note:
        This function assumes that the following lists are defined in the global scope:
            ho_bacillus, ho_pathogen, ho_interaction
        Overall is computed on the union of these sets.
    """
    # Validate metric and avg_mode choices.
    assert metric in [
        "RMSE",
        "MAE",
        "std_abs_err",
        "MAPE",
        "std_abs_relative_err",
        "R2",
    ], "metric must be one of: RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2"
    assert avg_mode in ["weighted", "unweighted"], (
        "avg_mode must be 'weighted' or 'unweighted'"
    )

    direction = "(Lower is better)" if metric != "R2" else "(Higher is better)"
    # Read the CSV file and filter by the selected method.
    results = pd.read_csv(results_file_path)
    results = results[results["Method"] == method]

    # Define the union of evaluation sets for overall performance.
    ho_all = ho_bacillus + ho_pathogen + ho_interaction

    # Helper function to compute aggregated statistics for a given subset.
    def agg_stats(df, eval_type):
        if avg_mode == "weighted":
            # Compute weighted mean using n_samples as weight, while using the unweighted std and count.
            grp = (
                df.groupby("Model")
                .apply(
                    lambda g: pd.Series(
                        {
                            "mean_metric": np.sum(g[metric] * g["n_samples"])
                            / np.sum(g["n_samples"]),
                        }
                    )
                )
                .reset_index()
            )
        else:
            # Compute unweighted mean.
            grp = (
                df.groupby("Model")
                .agg(
                    mean_metric=(metric, "mean"),
                )
                .reset_index()
            )

        # Compute confidence intervals using compute_CI
        if avg_mode == "weighted":
            ci_data = (
                results.groupby(["Method", "Model"])
                .apply(
                    lambda g: pd.Series(
                        compute_CI(
                            g[metric] * g["n_samples"] / np.sum(g["n_samples"]),
                            mode="bca",
                            stat_func=weighted_stat_func,
                        )
                    )
                )
                .rename(columns={0: "ci_low", 1: "ci_up"})
                .reset_index()
            )
        else:
            ci_data = (
                results.groupby(["Method", "Model"])
                .apply(lambda g: pd.Series(compute_CI(g[metric].values, mode="bca")))
                .rename(columns={0: "ci_low", 1: "ci_up"})
                .reset_index()
            )

        # Merge dataframes
        grp = pd.merge(grp, ci_data, on=["Model"])
        grp["ci"] = abs(grp["mean_metric"] - grp["ci_low"])

        grp["EvalType"] = eval_type
        return grp

    # Compute statistics for each evaluation type.
    overall_df = agg_stats(results[results["Evaluation"].isin(ho_all)], "Overall")
    bacillus_df = agg_stats(
        results[results["Evaluation"].isin(ho_bacillus)], "Bacillus"
    )
    pathogen_df = agg_stats(
        results[results["Evaluation"].isin(ho_pathogen)], "Pathogen"
    )
    interaction_df = agg_stats(
        results[results["Evaluation"].isin(ho_interaction)], "Interaction"
    )

    # Combine the aggregated data.
    agg_data = pd.concat(
        [overall_df, bacillus_df, pathogen_df, interaction_df], ignore_index=True
    )

    # Define the desired order for evaluation types and for models.
    eval_order = ["Overall", "Bacillus", "Pathogen", "Interaction"]
    order_models = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LinearSVR",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "LGBMRegressor",
        "XGBRegressor",
    ]
    available_models = [m for m in order_models if m in agg_data["Model"].unique()]

    assert set(agg_data["Model"].unique()).intersection(set(available_models)) == set(
        agg_data["Model"].unique()
    ), "We are missing models when defining order for plotting"
    # Ensure EvalType is categorical with the desired order.
    agg_data["EvalType"] = pd.Categorical(
        agg_data["EvalType"], categories=eval_order, ordered=True
    )
    # Sort the data by Model and EvalType.
    agg_data = agg_data.sort_values(["Model", "EvalType"])

    # Set publication-grade style.
    sns.set_theme(style="whitegrid", font="serif", font_scale=1)
    plt.rcParams.update(
        {"font.size": 8, "font.family": "sans-serif", "font.weight": "bold"}
    )

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(15, 7))

    # Draw the grouped barplot.
    ax = sns.barplot(
        data=agg_data,
        x="Model",
        y="mean_metric",
        hue="EvalType",
        palette="inferno",
        ax=ax,
        errorbar=None,
        order=available_models,
        hue_order=eval_order,
    )

    # Overlay error bars based on the computed 95% CI.
    for patch, ci in zip(ax.patches, agg_data["ci"].values):
        x_center = patch.get_x() + patch.get_width() / 2.0
        height = patch.get_height()
        ax.errorbar(x_center, height, yerr=ci, color="black", capsize=5, fmt="none")

    # Compute the overall best (lowest) value using the Overall group.
    if not overall_df.empty:
        best_val_series = overall_df["mean_metric"]
        best_val = best_val_series.min()
        best_model_series = overall_df.loc[
            overall_df["mean_metric"] == best_val, "Model"
        ]
        best_model = best_model_series.iloc[0] if not best_model_series.empty else "N/A"
    else:
        best_val = None
        best_model = "N/A"

    # Add a horizontal dotted line for the best overall value.
    if best_val is not None:
        ax.axhline(
            y=best_val,
            linestyle=":",
            color="red",
            linewidth=2,
            label=f"overall best: {best_val:.3f}",
        )

    # Set title and axis labels.
    ax.set_title(
        f"Summary Model Selection Performance ({metric})\n"
        f"Method: {method.upper()} | Averaging: {avg_mode.capitalize()} | Best Model (Overall): {best_model}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{metric} {direction}", fontsize=14, fontweight="bold")

    # Append custom legend entry for the overall
    # Append custom legend entry for the overall best line.
    handles, labels = ax.get_legend_handles_labels()
    if best_val is not None:
        best_handle = Line2D(
            [],
            [],
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"overall best: {best_val:.3f}",
        )
        if f"overall best: {best_val:.3f}" not in labels:
            handles.append(best_handle)
            labels.append(f"overall best: {best_val:.3f}")
    ax.legend(
        handles=handles,
        labels=labels,
        title="Evaluation Type",
        title_fontsize=12,
        fontsize=10,
    )

    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith(".pdf"):
            save_path = (
                save_path[: save_path.index(".pdf")]
                + f"_{method}_{avg_mode}"
                + save_path[save_path.index(".pdf") :]
            )
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path + f"_{method}_{avg_mode}" + ".pdf",
                format="pdf",
                bbox_inches="tight",
            )

    if show:
        plt.show()


def summary_preprocess_selection(
    results_file_path,
    metric="RMSE",
    method="avg",
    avg_mode="unweighted",
    save_path=None,
    save_agg_data=True,
    ci_normalized=False,
    show=False,
):
    """
    Produit un graphique résumant pour chaque modèle de preprocessing (par exemple "MeanImputer_StandardScaler")
    un point positionné à la valeur moyenne de la métrique (ex: RMSE) avec, en barre d'erreur, son intervalle de confiance à 95%.

    Les points sont tracés pour chacune des évaluations: Overall, Bacillus, Pathogen et Interaction.
    Le choix de la meilleure stratégie se fera en sélectionnant celle ayant le plus petit ratio (métrique / CI) dans le groupe Overall.

    Parameters:
        results_file_path : str
            Chemin vers le fichier CSV contenant les résultats.
        metric : str
            Une des valeurs parmi ["RMSE", "MAE", "std_abs_err", "MAPE", "std_abs_relative_err", "R2"].
        method : str
            La méthode sur laquelle filtrer (ex: "avg", "random", "combinatoric").
        avg_mode : str
            Mode d'agrégation, soit "weighted" (pondéré par n_samples) ou "unweighted" (par défaut).
        save_path : str, optionnel
            Si fourni, le graphique sera sauvegardé au format PDF.
        save_agg_data : bool, optionnel
            Si True, les données agrégées seront également sauvegardées au format CSV.
        ci_normalized : bool, optionnel
            Si True, la CI est calculée en normalisant par la valeur moyenne.
        show : bool, optionnel
            Si True, le graphique est affiché.
    """
    # Validation de la métrique et du mode d'agrégation
    assert metric in [
        "RMSE",
        "MAE",
        "std_abs_err",
        "MAPE",
        "std_abs_relative_err",
        "R2",
    ], (
        "La métrique doit être l'une de: RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2"
    )
    assert avg_mode in ["weighted", "unweighted"], (
        "avg_mode doit être 'weighted' ou 'unweighted'"
    )

    # Lecture du CSV et filtrage selon la méthode
    results = pd.read_csv(results_file_path)
    results = results[results["Method"] == method]

    # Ensemble des évaluations pour le groupe Overall
    ho_all = ho_bacillus + ho_pathogen + ho_interaction

    # Fonction d'agrégation pour un sous-ensemble d'évaluations.
    def agg_stats(df, eval_type, ci_normalized):
        if avg_mode == "weighted":
            # Calcul de la moyenne pondérée par n_samples, en conservant std et count non pondérés
            grp = (
                df.groupby("Model")
                .apply(
                    lambda g: pd.Series(
                        {
                            "mean_metric": np.sum(g[metric] * g["n_samples"])
                            / np.sum(g["n_samples"]),
                        }
                    )
                )
                .reset_index()
            )
            # Correction de l'erreur : suppression des colonnes dupliquées (ici, "Model")
            grp = grp.loc[:, ~grp.columns.duplicated()]
        else:
            # Calcul de la moyenne non pondérée.
            grp = (
                df.groupby("Model")
                .agg(
                    mean_metric=(metric, "mean"),
                )
                .reset_index()
            )
        # Compute confidence intervals using compute_CI
        if avg_mode == "weighted":
            ci_data = (
                results.groupby(["Method", "Model"])
                .apply(
                    lambda g: pd.Series(
                        compute_CI(
                            g[metric] * g["n_samples"] / np.sum(g["n_samples"]),
                            mode="bca",
                            stat_func=weighted_stat_func,
                        )
                    )
                )
                .rename(columns={0: "ci_low", 1: "ci_up"})
                .reset_index()
            )
        else:
            ci_data = (
                results.groupby(["Method", "Model"])
                .apply(lambda g: pd.Series(compute_CI(g[metric].values, mode="bca")))
                .rename(columns={0: "ci_low", 1: "ci_up"})
                .reset_index()
            )

        # Merge dataframes
        grp = pd.merge(grp, ci_data, on=["Model"])
        grp["ci"] = abs(grp["mean_metric"] - grp["ci_low"])

        grp["EvalType"] = eval_type
        return grp

    print(results)
    # Calcul des statistiques agrégées pour chaque groupe d'évaluation.
    overall_df = agg_stats(
        results[results["Evaluation"].isin(ho_all)],
        "Overall",
        ci_normalized=ci_normalized,
    )
    bacillus_df = agg_stats(
        results[results["Evaluation"].isin(ho_bacillus)],
        "Bacillus",
        ci_normalized=ci_normalized,
    )
    pathogen_df = agg_stats(
        results[results["Evaluation"].isin(ho_pathogen)],
        "Pathogen",
        ci_normalized=ci_normalized,
    )
    interaction_df = agg_stats(
        results[results["Evaluation"].isin(ho_interaction)],
        "Interaction",
        ci_normalized=ci_normalized,
    )

    # Concaténation des données agrégées
    agg_data = pd.concat(
        [overall_df, bacillus_df, pathogen_df, interaction_df], ignore_index=True
    )

    # Ordre souhaité pour les groupes d'évaluation et pour les modèles.
    eval_order = ["Overall", "Bacillus", "Pathogen", "Interaction"]
    order_models = []
    for imputer in [
        "MeanImputer",
        "MedianImputer",
        "KNNImputer",
        "RandomForestImputer",
    ]:
        for scaler in [
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "MaxAbsScaler",
        ]:
            order_models.append(f"{imputer}_{scaler}")
    available_models = [m for m in order_models if m in agg_data["Model"].unique()]

    assert set(agg_data["Model"].unique()).intersection(set(available_models)) == set(
        agg_data["Model"].unique()
    ), "We are missing models when defining order for plotting"

    agg_data["EvalType"] = pd.Categorical(
        agg_data["EvalType"], categories=eval_order, ordered=True
    )
    agg_data = agg_data.sort_values(["Model", "EvalType"])

    # Calcul de la meilleure stratégie en utilisant le ratio = mean_metric / ci sur le groupe Overall
    overall_group = overall_df.copy()
    overall_group = overall_group[
        overall_group["ci"] != 0
    ]  # éviter la division par zéro
    overall_group["ratio"] = overall_group["mean_metric"] / overall_group["ci"]
    if not overall_group.empty:
        best_idx = overall_group["ratio"].idxmax()
        best_model = overall_group.loc[best_idx, "Model"]
        best_metric = overall_group.loc[best_idx, "mean_metric"]
        best_ratio = overall_group.loc[best_idx, "ratio"]
    else:
        best_model = "N/A"
        best_metric = None
        best_ratio = None

    # Configuration du style du graphique
    sns.set_theme(style="whitegrid", font="serif", font_scale=1)
    plt.rcParams.update(
        {"font.size": 8, "font.family": "sans-serif", "font.weight": "bold"}
    )
    # Taille adaptée pour une orientation verticale
    fig, ax = plt.subplots(figsize=(7, 15))

    # Sauvegarde des données agrégées si demandé.
    if save_agg_data:
        if save_path is not None:
            if save_path.endswith(".pdf"):
                save_agg_path = (
                    save_path[: save_path.index(".pdf")]
                    + f"_{method}_{avg_mode}_agg_data.csv"
                )
            else:
                save_agg_path = save_path + f"_{method}_{avg_mode}_agg_data.csv"
            agg_data.to_csv(save_agg_path, index=False)

    # Pour un plot vertical, les modèles seront sur l'axe y.
    # On définit des offsets pour décaler les points par type d'évaluation (pour éviter qu'ils ne se superposent)
    dodge_offsets = {
        "Overall": -0.15,
        "Bacillus": -0.05,
        "Pathogen": 0.05,
        "Interaction": 0.15,
    }
    # Mapping des modèles aux positions sur l'axe y
    model_to_y = {model: i for i, model in enumerate(available_models)}

    # Pour chaque type d'évaluation, tracer le point et son intervalle de confiance.
    for eval_type in eval_order:
        sub = agg_data[agg_data["EvalType"] == eval_type]
        # Calcul des positions y décalées
        y_positions = [model_to_y[m] + dodge_offsets[eval_type] for m in sub["Model"]]
        ax.errorbar(
            sub["mean_metric"],  # x : valeur de la métrique
            y_positions,  # y : position sur l'axe des modèles
            xerr=sub["ci"],  # barres d'erreur horizontales
            fmt="o",
            capsize=5,
            markersize=8,
            label=eval_type,
            linestyle="None",
            markeredgecolor="black",
        )

    # Ajout d'une ligne verticale pointillée indiquant la meilleure stratégie (selon le ratio)
    if best_metric is not None:
        ax.axvline(
            x=best_metric,
            linestyle=":",
            color="red",
            linewidth=2,
            label=f"{metric} : {best_metric:.3f}",
        )

    ax.set_title(
        f"Summary of preprocessing selection ({metric} ± 95% CI)\n"
        f"Method: {method.upper()} | Averaging: {avg_mode.capitalize()}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Preprocessing", fontsize=14, fontweight="bold")
    ax.set_xlabel(metric, fontsize=9, fontweight="bold")

    # Positionnement et étiquetage de l'axe des y
    ax.set_yticks(range(len(available_models)))
    ax.set_yticklabels(available_models, rotation=0)

    # Place legend outside (below the plot)
    ax.legend(
        title="Evaluation type",
        title_fontsize=12,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=1,
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith(".pdf"):
            save_path_modified = (
                save_path[: save_path.index(".pdf")]
                + f"_{method}_{avg_mode}"
                + save_path[save_path.index(".pdf") :]
            )
            plt.savefig(save_path_modified, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path + f"_{method}_{avg_mode}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

    if show:
        plt.show()


def plot_native_feature_selection(
    path_df=None, ci_mode="bca", save_path=None, show=False
):
    """
    Plot the Permutation Feature Importance (PFI) with 95% Confidence Intervals
    for all step_i files in the specified folder.

    Parameters:
    - folder_path (str, optional): Path to folder containing CSV step files.
      Defaults to "./Results/native_feature_selection/".
    - ci_mode (str): Mode for computing confidence intervals (default: "bca").
    - save_path (str, optional): File path (without extension) to save the figure as PDF.
    - show (bool): Whether to display the plot.
    """
    if path_df is None:
        folder = "./Results/native_feature_selection/"
    elif os.path.isdir(path_df):
        folder = path_df
    else:
        folder = None

    if folder is not None:
        # Look for step_i files in the folder.
        file_pattern = os.path.join(folder, "step_*_permutation_details.csv")
        file_list = sorted(glob.glob(file_pattern))
        summary_pattern = os.path.join(folder, "step_*_permutation_results.csv")
        summaries = sorted(glob.glob(summary_pattern))
    else:
        file_list = [path_df]

    if not file_list:
        print("No step files found in the specified path.")
        return

    n_steps = len(file_list)
    # Create a subplot for each step file.
    fig, axes = plt.subplots(n_steps, 1, figsize=(10, 6 * n_steps), squeeze=False)
    axes = axes.flatten()

    sns.set_theme(style="whitegrid")
    # print(file_list)

    baseline_info = []
    for idx, file in enumerate(summaries):
        summary_step = pd.read_csv(file)
        w_rmse = summary_step["Weighted RMSE"] - summary_step["Weighted diff_RMSE"]
        w_mae = summary_step["Weighted MAE"] - summary_step["Weighted diff_MAE"]
        baseline_info.append(0.5 * (w_mae.iloc[0] + w_rmse.iloc[0]))

    baseinfo_idx = 0
    for idx, file in enumerate(file_list):
        try:
            # Read the permutation details for the step.
            step_df = pd.read_csv(file)
            # print(step_df.columns)
            # Initialize dictionary to accumulate plotting data.
            plot_data = {"Feature": [], "PFI": [], "CI95_low": [], "CI95_up": []}

            # Compute PFI and confidence intervals for each candidate feature.
            for permutation in pd.unique(step_df["Permutation"]):
                if permutation != "No Permutation":
                    mask = step_df["Permutation"] == permutation
                    # The column "Weighted Cross Mean" is assumed to contain the weighted contributions.
                    avg = step_df[mask]["Weighted Cross Mean"].sum()
                    # print(permutation, avg)
                    if avg != 0:
                        low, up = compute_CI(
                            step_df[mask]["Weighted Cross Mean"],
                            num_iter=5000,
                            confidence=95,
                            seed=62,
                            mode=ci_mode,
                            stat_func=weighted_stat_func,
                        )
                        # Determine the size (offset) of the confidence interval relative to the mean.
                        low, up = abs(avg - low), abs(up - avg)
                    else:
                        avg, low, up = 0, 0, 0

                    plot_data["Feature"].append(permutation)
                    plot_data["PFI"].append(avg)
                    plot_data["CI95_low"].append(low)
                    plot_data["CI95_up"].append(up)

            # Convert to DataFrame and sort by PFI (descending order).
            plot_df = pd.DataFrame(plot_data).sort_values("PFI", ascending=False)

            ax = axes[idx]
            # Create horizontal bar plot.
            sns.barplot(
                data=plot_df,
                y="Feature",
                x="PFI",
                orient="h",
                hue="Feature",
                palette="magma_r",
                edgecolor="black",
                ax=ax,
            )

            # Plot the confidence intervals.
            intervals = np.array([plot_df["CI95_low"], plot_df["CI95_up"]])
            ax.errorbar(
                y=np.arange(plot_df.shape[0]),
                x=plot_df["PFI"],
                xerr=intervals,
                fmt="o",
                capsize=4,
                elinewidth=1.5,
                color="black",
                alpha=0.8,
            )

            # Set axis labels and title. Extract step name from the filename.
            step_name = os.path.basename(file).split("_permutation_details.csv")[0]
            ax.set_xlabel(
                "Permutation Feature Importance (PFI)", fontsize=14, fontweight="bold"
            )
            ax.set_ylabel("Feature", fontsize=14, fontweight="bold")
            ax.set_title(
                f"PFI with 95% CI - Step {baseinfo_idx + 1} - Baseline Cross Mean: {baseline_info[baseinfo_idx]:.3f}",
                fontsize=16,
                fontweight="bold",
            )

            # Annotate each bar with its PFI value.
            for i, patch in enumerate(ax.patches):
                y_center = patch.get_y() + patch.get_height() / 2.0
                x_top = patch.get_width() + intervals[1, i]
                label = f"{patch.get_width():.3f}"
                ax.text(
                    x_top + 0.005,
                    y_center - 0.005,
                    label,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )
            baseinfo_idx += 1
        except Exception as e:
            print(f"An error occurred at step {idx + 1}")
            print(e)
            print(traceback.format_exc())

    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


# Suuggère que P SubCov et B_ROughness et P volume sont pas pertinents
# Entraine modèle sans ses 3 d'un coup, sans chacun d'entre eux, sans pairs
# Compare à baseline, si on gagne en perf alors c'est à envlever.
# SI pas signif reste comme ça


def plot_feature_engineering(path_df=None, ci_mode="bca", save_path=None, show=False):
    """
    Plots feature engineering results for each step_i file found in the given folder.
    For each file, the function generates:
    1. A bar plot for the top 10 features sorted by PFI (Permutation Feature Importance),
       with error bars for 95% confidence intervals.
    2. A heatmap displaying RMSE, MAE, and PFI across all features.

    Parameters:
    - path_df (str, optional): Either a path to a specific CSV file or a folder containing
      step_i CSV files. If None, defaults to "./Results/feature_engineering/".
    - ci_mode (str): Mode for computing confidence intervals (default: "bca").
    - save_path (str, optional): Base path (without extension) to save the plots as PDF.
      The step identifier is appended to the filename.
    - show (bool): Whether to display the plots.
    """
    # Determine if we are given a folder or a single file.
    if path_df is None:
        folder = "./Results/feature_engineering/"
    elif os.path.isdir(path_df):
        folder = path_df
    else:
        folder = None

    if folder is not None:
        # Look for step_i files in the folder.
        file_pattern = os.path.join(folder, "step_*_permutation_details.csv")
        file_list = sorted(glob.glob(file_pattern))
        summary_pattern = os.path.join(folder, "step_*_permutation_results.csv")
        summaries = sorted(glob.glob(summary_pattern))
    else:
        file_list = [path_df]

    if not file_list:
        print("No step files found in the specified path.")
        return

    baseline_info = []
    for idx, file in enumerate(summaries):
        summary_step = pd.read_csv(file)
        w_rmse = summary_step["Weighted RMSE"] - summary_step["Weighted diff_RMSE"]
        w_mae = summary_step["Weighted MAE"] - summary_step["Weighted diff_MAE"]
        baseline_info.append(0.5 * (w_mae.iloc[0] + w_rmse.iloc[0]))

    baseinfo_idx = 0
    # Process each step_i file.
    for idx, file in enumerate(file_list):
        try:
            # Load results from the current step file.
            step_df = pd.read_csv(file)

            plot_data = {"Feature": [], "PFI": [], "CI95_low": [], "CI95_up": []}

            # Compute PFI and confidence intervals for each candidate feature.
            for permutation in pd.unique(step_df["Permutation"]):
                if permutation != "No Permutation":
                    mask = step_df["Permutation"] == permutation
                    # The column "Weighted Cross Mean" is assumed to contain the weighted contributions.
                    avg = step_df[mask]["Weighted Cross Mean"].sum()
                    # print(permutation, avg)
                    if avg != 0:
                        low, up = compute_CI(
                            step_df[mask]["Weighted Cross Mean"],
                            num_iter=5000,
                            confidence=95,
                            seed=62,
                            mode=ci_mode,
                            stat_func=weighted_stat_func,
                        )
                        # Determine the size (offset) of the confidence interval relative to the mean.
                        low, up = abs(avg - low), abs(up - avg)
                    else:
                        avg, low, up = 0, 0, 0

                    plot_data["Feature"].append(permutation)
                    plot_data["PFI"].append(avg)
                    plot_data["CI95_low"].append(low)
                    plot_data["CI95_up"].append(up)

            # Convert to DataFrame and sort by PFI (descending order).
            plot_df = pd.DataFrame(plot_data).sort_values("PFI", ascending=False)

            # -----------------------
            # Bar Plot for Top 10 Features
            # -----------------------
            top_features = plot_df.iloc[:10]

            sns.set_theme(style="whitegrid")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=top_features,
                y="Feature",
                x="PFI",
                orient="h",
                palette="magma_r",
                edgecolor="black",
                ax=ax_bar,
            )
            # Add error bars for confidence intervals.
            intervals = np.array([top_features["CI95_low"], top_features["CI95_up"]])
            ax_bar.errorbar(
                y=np.arange(top_features.shape[0]),
                x=top_features["PFI"],
                xerr=intervals,
                fmt="o",
                capsize=4,
                elinewidth=1.5,
                color="black",
                alpha=0.8,
            )
            # Annotate each bar with its PFI value.
            for i, patch in enumerate(ax_bar.patches):
                y_center = patch.get_y() + patch.get_height() / 2.0
                x_top = patch.get_width() + intervals[1, i]
                label = f"{patch.get_width():.3f}"
                ax_bar.text(
                    x_top + 0.005,
                    y_center - 0.002,
                    label,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )
            ax_bar.set_xlabel(
                "Permutation Feature Importance (PFI)", fontsize=14, fontweight="bold"
            )
            ax_bar.set_ylabel("Feature", fontsize=14, fontweight="bold")
            step_name = os.path.basename(file).split("_permutation_details.csv")[0]

            ax_bar.set_title(
                f"Top 10 Feature Importances with 95% CI - Step {baseinfo_idx + 1} - Baseline Cross Mean: {baseline_info[baseinfo_idx]:.3f}",
                fontsize=16,
                fontweight="bold",
            )

            fig_bar.tight_layout()
            if save_path is not None:
                save_bar = (
                    f"{save_path}_step_{baseinfo_idx + 1}_top.pdf"
                    if not save_path.endswith(".pdf")
                    else f"{save_path}_step_{baseinfo_idx + 1}_top"
                )
                fig_bar.savefig(save_bar, format="pdf", bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig_bar)

            # -----------------------
            # Heatmap for All Features
            # -----------------------
            # Set feature names as index.
            plot_df.set_index("Feature", inplace=True)
            # Drop features if present.
            for drop_feature in ["prod_II_III", "prod_I_III", "prod_I_II"]:
                if drop_feature in plot_df.index:
                    plot_df.drop(index=drop_feature, inplace=True)
            fig_heat = plt.figure(figsize=(12, 26))
            sns.heatmap(
                plot_df[["PFI", "CI95_low", "CI95_up"]],
                cmap="mako",
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                cbar_kws={"shrink": 0.75},
            )

            plt.title(
                f"Feature Engineering Metrics Heatmap - Step {baseinfo_idx + 1}",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            if save_path is not None:
                save_heat = (
                    f"{save_path}_step_{baseinfo_idx + 1}_heat.pdf"
                    if not save_path.endswith(".pdf")
                    else f"{save_path}_step_{baseinfo_idx + 1}_heat"
                )
                fig_heat.savefig(save_heat, format="pdf", bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig_heat)
            baseinfo_idx += 1
        except Exception as e:
            print(f"An error occurred at step {idx + 1} (file: {file})")
            print(e)


# Refaire sans les produits entre Modalités de modele, remplace les ratios par les soustractions,
# vérifie fonction puissance


def plot_optuna_study(path_study=None, save_path=None, show=False):
    """
    Plots the results of an Optuna study, including:
    1. Parameter importances (bar plot)
    2. Parameter rankings (rank plot)

    Parameters:
    - path_study (str, optional): Path to the saved Optuna study (Pickle file).
      If None, uses a default path.
    """

    # Load Optuna study
    if path_study is None:
        path_study = "./Results/optuna_campaign/optuna_study.pkl"

    with open(path_study, "rb") as f:
        optuna_study = pkl.load(f)

    # Set seaborn theme for consistent styling
    sns.set_theme(style="whitegrid")

    # Plot parameter importances
    plot_param_importances(optuna_study)
    plt.gca().set_title("", loc="left")
    plt.gca().set_title("Optuna Parameter Importances", fontsize=14, fontweight="bold")
    plt.gca().set_ylabel("Hyperparameter", fontsize=11, fontweight="bold")
    plt.gca().set_xlabel("Hyperparameter Importance", fontsize=11, fontweight="bold")
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path_bis = save_path + "_importances.pdf"
            plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    # Plot parameter ranking
    plot_rank(
        optuna_study,
        params=[
            "n_estimators",
            "num_leaves",
            "bagging_fraction",
            "feature_fraction",
            "min_child_samples",
        ],
    )
    plt.gcf().set_figheight(20)
    plt.gcf().set_figwidth(15)
    plt.suptitle("Optuna Parameter Rankings", fontsize=14, fontweight="bold")

    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path_bis = save_path + "_rank.pdf"
            plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def plot_feature_importance_heatmap(path_model_folder=None, save_path=None, show=False):
    """
    Plots a heatmap of feature importances across multiple hold-out folds
    from LightGBM models trained and stored in a specified folder.

    Parameters:
    - path_model_folder (str, optional): Path to the folder containing saved LightGBM models.
      If None, uses a default path.
    """

    # Define base model name and default path
    base_model_name = "LGBMRegressor"
    if path_model_folder is None:
        path_model_folder = "./Results/models/"

    # Retrieve model files
    file_list = [
        os.path.join(path_model_folder, f) for f in os.listdir(path_model_folder)
    ]
    file_list = [f for f in file_list if f.endswith(".txt")]  # Ensure only model files

    if not file_list:
        print("No valid LightGBM model files found in the directory.")
        return

    # Initialize DataFrame storage
    plot_df = {"Hold-Out Fold": []}

    # Extract feature names from the first model
    model = lgb.Booster(model_file=file_list[0])
    features = model.feature_name()

    for feature in features:
        plot_df[feature] = []

    # Collect feature importances for each model
    for model_path in file_list:
        model = lgb.Booster(model_file=model_path)
        feat_imp = model.feature_importance("gain")

        # Extract hold-out fold identifier from filename
        hold_out_name = model_path[
            len(path_model_folder) + len(base_model_name) + 1 : model_path.index(
                "_model"
            )
        ]
        plot_df["Hold-Out Fold"].append(hold_out_name)
        for i, feature in enumerate(features):
            plot_df[feature].append(feat_imp[i])

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_df)

    # Set Seaborn theme for consistency
    sns.set_theme(style="whitegrid")

    # Plot heatmap
    plt.figure(figsize=(12, 26))  # Adjust figure size for better readability
    sns.heatmap(
        plot_df.set_index("Hold-Out Fold"),
        cmap="viridis",
        linewidths=0.5,
        linecolor="gray",
        # annot=True,  # Show values in heatmap
        fmt=".3f",
    )

    # Titles and labels
    plt.title(
        "Feature Importance (Gain) Across Hold-Out Folds",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Features", fontsize=12, fontweight="bold")
    plt.ylabel("Hold-Out Fold", fontsize=12, fontweight="bold")

    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


ablations = {
    "None": "None removed",
    "all_B": "(-) Bacillus Parameters",
    "all_P": "(-) Pathogen Parameters",
    "B_Biofilm_Height": "(-) B_Biofilm_Height",
    "B_Biofilm_Volume": "(-) B_Biofilm_Volume",
    "B_Biofilm_Roughness": "(-) B_Biofilm_Roughness",
    "B_Biofilm_SubstratumCoverage": "(-) B_Biofilm_SubstratumCoverage",
    "P_Biofilm_Height": "(-) P_Biofilm_Height",
    "P_Biofilm_Volume": "(-) P_Biofilm_Volume",
    "P_Biofilm_Roughness": "(-) P_Biofilm_Roughness",
    "P_Biofilm_SubstratumCoverage": "(-) P_Biofilm_SubstratumCoverage",
    "Modele": "(-) Modele",
}


def check_isin(text):
    """Returns the ablation key if it exists in the filename, else None."""
    for key in ablations.keys():
        if key in text:
            return ablations[key]  # Return human-readable name
    return None


def weighted_stat_func(data):
    """Computes a weighted sum for confidence interval calculations."""
    return np.sum(data)


def plot_ablation_study(
    path_ablation_folder=None, weighted=True, ci_mode="bca", save_path=None, show=False
):
    """
    Plots an ablation study barplot with error bars for RMSE & MAE.
    The baseline ("None removed") is always shown first.
    Ablations with an increased metric relative to the baseline are colored
    with a red-ish gradient, while those with a decreased metric use a green-ish gradient.
    Black contours are added to each bar.
    Bar labels (with two decimals) are added at the center of each bar;
    the label text color is chosen based on the bar brightness (black for bright, white for dark).

    Parameters:
    - path_ablation_folder (str, optional): Path to the folder containing ablation results.
    - weighted (bool): Whether to weight RMSE/MAE by the number of samples.
    - ci_mode (str): Confidence interval calculation method.
    - save_path (str, optional): If provided, saves the figure as a PDF.
    - show (bool): If True, displays the plots.
    """
    # Retrieve files, excluding ho_all_results.csv
    if path_ablation_folder is None:
        path_ablation_folder = "./Results/ablation_study/"
    file_list = os.listdir(path_ablation_folder)
    if "ho_all_results.csv" in file_list:
        file_list.remove("ho_all_results.csv")

    # Map file names to human-readable ablation names using check_isin
    keys = [check_isin(file) for file in file_list]

    # Initialize plot DataFrame
    addon = "Weighted " if weighted else ""
    plot_df = {
        "Features": [],
        addon + "RMSE": [],
        addon + "MAE": [],
        "RMSE_CI95_low": [],
        "RMSE_CI95_up": [],
        "MAE_CI95_low": [],
        "MAE_CI95_up": [],
    }

    # Process each ablation file
    for i, ablation in enumerate(file_list):
        if keys[i] is not None:
            df = pd.read_csv(os.path.join(path_ablation_folder, ablation))
            # Compute weighted or mean RMSE & MAE
            if weighted:
                df[addon + "RMSE"] = (
                    df["RMSE"] * df["n_samples"] / df["n_samples"].sum()
                )
                df[addon + "MAE"] = df["MAE"] * df["n_samples"] / df["n_samples"].sum()
                rmse, mae = df[addon + "RMSE"].sum(), df[addon + "MAE"].sum()
            else:
                rmse, mae = df["RMSE"].mean(), df["MAE"].mean()

            plot_df["Features"].append(keys[i])
            plot_df[addon + "MAE"].append(mae)
            plot_df[addon + "RMSE"].append(rmse)

            # Compute confidence intervals
            func = weighted_stat_func if weighted else stat_func
            # RMSE CI
            low_r, up_r = compute_CI(
                df[addon + "RMSE"],
                num_iter=5000,
                confidence=95,
                seed=62,
                stat_func=func,
                mode=ci_mode,
            )
            avg_r = func(df[addon + "RMSE"])
            low_r, up_r = abs(avg_r - low_r), abs(up_r - avg_r)
            # MAE CI
            low_m, up_m = compute_CI(
                df[addon + "MAE"],
                num_iter=5000,
                confidence=95,
                seed=62,
                stat_func=func,
                mode=ci_mode,
            )
            avg_m = func(df[addon + "MAE"])
            low_m, up_m = abs(avg_m - low_m), abs(up_m - avg_m)

            plot_df["RMSE_CI95_low"].append(low_r)
            plot_df["RMSE_CI95_up"].append(up_r)
            plot_df["MAE_CI95_low"].append(low_m)
            plot_df["MAE_CI95_up"].append(up_m)

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_df)

    # Set theme for a clean look
    sns.set_theme(style="whitegrid", context="talk")
    # For each metric, create bar plots with custom colors and bar labels.
    for metric in ["RMSE", "MAE"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Reorder so that baseline ("None removed") is first
        baseline_df = plot_df[plot_df["Features"] == "None removed"]
        others = plot_df[plot_df["Features"] != "None removed"].sort_values(
            by=addon + metric, ascending=False
        )
        df_plot = pd.concat([baseline_df, others], axis=0).reset_index(drop=True)
        # Get baseline value (assumed to be first row)
        baseline_value = df_plot.iloc[0][addon + metric]

        # Compute maximum differences for normalization
        pos_mask = df_plot[addon + metric] > baseline_value
        neg_mask = df_plot[addon + metric] < baseline_value
        max_positive = (
            (df_plot.loc[pos_mask, addon + metric] - baseline_value).max()
            if pos_mask.any()
            else 1
        )
        max_negative = (
            (baseline_value - df_plot.loc[neg_mask, addon + metric]).max()
            if neg_mask.any()
            else 1
        )

        # Build list of colors based on difference relative to baseline:
        # Baseline gets gray; above baseline: use Reds; below: use Greens.
        colors = []
        for _, row in df_plot.iterrows():
            if row["Features"] == "None removed":
                colors.append("gray")
            else:
                diff = row[addon + metric] - baseline_value
                if diff > 0:
                    norm_val = diff / max_positive if max_positive else 0
                    colors.append(cm.Reds(norm_val))
                elif diff < 0:
                    norm_val = (-diff) / max_negative if max_negative else 0
                    colors.append(cm.Greens(norm_val))
                else:
                    colors.append("gray")

        # Draw horizontal barplot with black edges
        barplot = sns.barplot(
            data=df_plot,
            y="Features",
            x=addon + metric,
            orient="h",
            ax=ax,
            palette=colors,
            edgecolor="black",
        )

        # Add error bars with CI
        intervals = np.array(
            [df_plot[f"{metric}_CI95_low"], df_plot[f"{metric}_CI95_up"]]
        )
        ax.errorbar(
            x=df_plot[addon + metric],
            y=np.arange(df_plot.shape[0]),
            xerr=intervals,
            capsize=5,
            fmt="o",
            ecolor="black",
        )

        # Add black contours (redundant if already set via edgecolor)
        for patch in ax.patches:
            patch.set_edgecolor("black")
            patch.set_linewidth(1)
            # Calculate center coordinates of each bar
            x_center = patch.get_x() + patch.get_width() / 2.0
            y_center = patch.get_y() + patch.get_height() / 2.0
            # Compute luminance of bar face (0-1 scale) using standard formula
            facecolor = patch.get_facecolor()  # RGBA tuple
            luminance = (
                0.299 * facecolor[0] + 0.587 * facecolor[1] + 0.114 * facecolor[2]
            )
            text_color = "black" if luminance > 0.5 else "white"
            label = f"{patch.get_width():.3f}"
            ax.text(
                x_center,
                y_center,
                label,
                color=text_color,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )

        ax.set_title(
            f"Ablation Study: Impact on {addon + metric}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel(f"{addon + metric} Score", fontsize=14, fontweight="bold")
        ax.set_ylabel("Ablated Feature", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path is not None:
            if not save_path.endswith(".pdf"):
                save_path_bis = save_path + f"_{metric}.pdf"
                plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
        if show:
            plt.show()


def str2arr(y_str):
    if y_str.startswith("[["):
        y_str = re.sub("\]\n \[", " ", y_str[2:-2])
        return np.fromstring(y_str, sep=" ")
    elif y_str.startswith("["):
        return np.fromstring(y_str[1:-1], sep=" ")
    else:
        return np.fromstring(y_str, sep=",")


def load_lgbm_model(path_model_folder=None, path_df=None, ho_name="1234_x_S.en"):
    from pipeline import create_pipeline

    if path_model_folder is None:
        path_model_folder = "./Results/models/"
    file_list = os.listdir(path_model_folder)
    file_list = [path_model_folder + file for file in file_list]
    model_file = [file for file in file_list if ho_name in file][0]

    model = lgb.Booster(model_file=model_file)

    if path_df is None:
        method_df = pd.read_csv("./Data/Datasets/fe_combinatoric_COI.csv")
    else:
        method_df = pd.read_csv(path_df)

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
        col for col in method_df.columns if col not in cat_cols + remove_cols + target
    ]

    estimator = create_pipeline(
        num_cols,
        cat_cols,
        imputer="KNNImputer",
        scaler="RobustScaler",
        estimator=model,
        model_name="LGBMRegressor",
    )
    return estimator, method_df


def retrieve_data(method_df, ho_name):
    ho_sets = get_hold_out_sets(
        "combinatoric", ho_folder_path="Data/Datasets/", suffix="_hold_outs.pkl"
    )
    X_train, X_test, Y_train, Y_test = get_train_test_split(
        ho_name,
        method_df,
        ho_sets,
    )
    return X_train, X_test, Y_train, Y_test


def check_empty(data, mask):
    if data[mask].shape[0] == 0:
        return True
    return False


def plot_err_distrib(path_df=None, ci_mode="bca", save_path=None, show=False):
    """
    Produces two plots:

    1. A horizontal bar plot showing the average percentage of predictions
       that fall under various absolute error thresholds, with error bars.
       - The bars follow a green colormap gradient: "< 0.01" is the darkest green,
         progressing to brighter green for "< 0.2". The ">= 0.2" bar is shown in red.

    2. A vertical bar plot showing the average absolute error by predicted exclusion
       score range, with error bars. The bar colors are set using a continuous colormap
       (with an accompanying colorbar representing the True Scores Proportion).
       A red horizontal reference line is drawn at 0.2 and labeled accordingly.

    Parameters:
    - path_df (str, optional): Path to CSV file containing ablation study results.
    - ci_mode (str): Confidence interval calculation method.
    """
    # ---------------------- Part 1: Distribution of Errors ---------------------- #
    if path_df is None:
        results = pd.read_csv(
            "./Results/ablation_study/ho_None_LGBMRegressor_results.csv"
        )
    else:
        results = pd.read_csv(path_df)

    # Compute percentage of predictions under various error thresholds for each Hold-Out Fold.
    plot_df = {
        "Hold-Out Fold": [],
        "< 0.01": [],
        "< 0.05": [],
        "< 0.1": [],
        "< 0.15": [],
        "< 0.2": [],
        ">= 0.2": [],
    }
    memory = {}
    for ho_name in results["Evaluation"]:
        try:
            method_df = pd.read_csv("./Data/Datasets/fe_combinatoric_COI.csv")

            file_list = os.listdir("./Results/models/")
            file_list = ["./Results/models/" + file for file in file_list]
            model_file = [file for file in file_list if ho_name in file][0]

            with open(model_file, "rb") as f:
                pipeline = pkl.load(f)

            X_train, X_test, _, y_true = retrieve_data(method_df, ho_name)

            X_test = pipeline[:-1].transform(X_test)  # preprocess X_test

            yhat = pipeline[-1].predict(X_test).reshape(-1, 1)
        except Exception as e:
            Warning("Using pickling failed, trying load_lgbm function..")
            pipeline, method_df = load_lgbm_model(
                "./Results/models/", "./Data/Datasets/fe_combinatoric_COI.csv", ho_name
            )
            X_train, X_test, _, y_true = retrieve_data(method_df, ho_name)
            pipeline[:-1].fit(X_train)
            X_test = pipeline[:-1].transform(X_test)

            yhat = pipeline[-1].predict(X_test).reshape(-1, 1)

        y_true = np.array(y_true).reshape(-1, 1)
        abs_err = np.abs(yhat - y_true)
        memory[ho_name] = {"yhat": yhat, "y_true": y_true}

        plot_df["Hold-Out Fold"].append(ho_name)
        plot_df["< 0.01"].append(np.mean(abs_err < 0.01))
        plot_df["< 0.05"].append(np.mean(abs_err < 0.05))
        plot_df["< 0.1"].append(np.mean(abs_err < 0.1))
        plot_df["< 0.15"].append(np.mean(abs_err < 0.15))
        plot_df["< 0.2"].append(np.mean(abs_err < 0.2))
        plot_df[">= 0.2"].append(np.mean(abs_err >= 0.2))
    plot_df = pd.DataFrame(plot_df)

    # Aggregate across folds: compute average and confidence intervals for each threshold.
    final_plot_df = {
        "Absolute Error": [],
        "Percentage Of Predictions": [],
        "CI95_low": [],
        "CI95_up": [],
    }
    for col in list(plot_df.columns)[1:]:
        avg = np.mean(plot_df[col])
        low, up = compute_CI(
            plot_df[col],
            num_iter=5000,
            confidence=95,
            seed=62,
            stat_func=stat_func,
            mode=ci_mode,
        )
        low, up = abs(avg - low), abs(up - avg)
        final_plot_df["Absolute Error"].append(col)
        final_plot_df["Percentage Of Predictions"].append(avg)
        final_plot_df["CI95_low"].append(low)
        final_plot_df["CI95_up"].append(up)
    final_plot_df = pd.DataFrame(final_plot_df)

    # Create custom colors:
    # For thresholds other than ">= 0.2", use a green gradient.
    # We use 5 shades from dark to light for "< 0.01" to "< 0.2".
    greens = sns.color_palette("Greens_r", 5)  # dark -> light
    color_mapping = {
        "< 0.01": greens[0],
        "< 0.05": greens[1],
        "< 0.1": greens[2],
        "< 0.15": greens[3],
        "< 0.2": greens[4],
        ">= 0.2": "red",
    }
    colors_plot1 = [color_mapping[val] for val in final_plot_df["Absolute Error"]]

    # Set Seaborn theme
    sns.set_theme(style="whitegrid", context="talk")

    # Plot 1: Horizontal bar plot for error distribution.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=final_plot_df,
        x="Absolute Error",
        y="Percentage Of Predictions",
        ax=ax,
        palette=colors_plot1,
    )
    intervals = np.array([final_plot_df["CI95_low"], final_plot_df["CI95_up"]])
    ax.errorbar(
        x=np.arange(final_plot_df.shape[0]),
        y=final_plot_df["Percentage Of Predictions"],
        yerr=intervals,
        capsize=5,
        fmt="_",
        markersize=10,
        ecolor="black",
    )
    for i, patch in enumerate(ax.patches):
        x_center = patch.get_x() + patch.get_width() / 2.0
        y_top = patch.get_height() + intervals[1, i]
        label = f"{patch.get_height():.3f}"
        ax.text(
            x_center,
            y_top + 0.005,
            label,
            ha="center",
            va="bottom",
            color="black",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_title("Distribution of Prediction Errors", fontsize=16, fontweight="bold")
    ax.set_xlabel("Error Threshold", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage of Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path_bis = save_path + "_distrib.pdf"
            plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    # ----------------- Part 2: Average Error vs. Predicted Score ----------------- #
    # Compute average absolute error and proportion for different predicted score ranges.
    plot_df2 = {
        "Hold-Out Fold": [],
        "< 0.2": [],
        "SIZE (< 0.2)": [],
        "[0.2,0.4[": [],
        "SIZE ([0.2,0.4[)": [],
        "[0.4,0.6[": [],
        "SIZE ([0.4,0.6[)": [],
        "[0.6,0.8[": [],
        "SIZE ([0.6,0.8[)": [],
        ">= 0.8": [],
        "SIZE (>= 0.8)": [],
    }
    for ho_name in results["Evaluation"]:
        yhat, y_true = memory[ho_name].values()
        abs_err = np.abs(yhat - y_true)
        _0_2 = (
            np.mean(abs_err[yhat < 0.2]) if not check_empty(abs_err, yhat < 0.2) else 0
        )
        n_0_2 = 0 if _0_2 == 0 else np.mean(yhat < 0.2)
        _0_4 = (
            np.mean(abs_err[(yhat >= 0.2) & (yhat < 0.4)])
            if not check_empty(abs_err, (yhat >= 0.2) & (yhat < 0.4))
            else 0
        )
        n_0_4 = 0 if _0_4 == 0 else np.mean((yhat >= 0.2) & (yhat < 0.4))
        _0_6 = (
            np.mean(abs_err[(yhat >= 0.4) & (yhat < 0.6)])
            if not check_empty(abs_err, (yhat >= 0.4) & (yhat < 0.6))
            else 0
        )
        n_0_6 = 0 if _0_6 == 0 else np.mean((yhat >= 0.4) & (yhat < 0.6))
        _0_8 = (
            np.mean(abs_err[(yhat >= 0.6) & (yhat < 0.8)])
            if not check_empty(abs_err, (yhat >= 0.6) & (yhat < 0.8))
            else 0
        )
        n_0_8 = 0 if _0_8 == 0 else np.mean((yhat >= 0.6) & (yhat < 0.8))
        sup_0_8 = (
            np.mean(abs_err[yhat >= 0.8])
            if not check_empty(abs_err, yhat >= 0.8)
            else 0
        )
        n_sup_0_8 = 0 if sup_0_8 == 0 else np.mean(yhat >= 0.8)

        plot_df2["Hold-Out Fold"].append(ho_name)
        plot_df2["< 0.2"].append(_0_2)
        plot_df2["SIZE (< 0.2)"].append(n_0_2)
        plot_df2["[0.2,0.4["].append(_0_4)
        plot_df2["SIZE ([0.2,0.4[)"].append(n_0_4)
        plot_df2["[0.4,0.6["].append(_0_6)
        plot_df2["SIZE ([0.4,0.6[)"].append(n_0_6)
        plot_df2["[0.6,0.8["].append(_0_8)
        plot_df2["SIZE ([0.6,0.8[)"].append(n_0_8)
        plot_df2[">= 0.8"].append(sup_0_8)
        plot_df2["SIZE (>= 0.8)"].append(n_sup_0_8)
    plot_df2 = pd.DataFrame(plot_df2)

    final_plot_df2 = {
        "Absolute Error": [],
        "True Scores Proportion": [],
        "Predicted Exclusion Score Range": [],
        "CI95_low": [],
        "CI95_up": [],
    }
    for i in range(1, plot_df2.shape[1]):
        col_name = list(plot_df2.columns)[i]
        if not col_name.startswith("SIZE"):
            print(plot_df2.iloc[:, i])
            avg = np.mean(plot_df2.iloc[:, i])
            if avg != 0:
                low, up = compute_CI(
                    plot_df2.iloc[:, i],
                    num_iter=5000,
                    confidence=95,
                    seed=62,
                    stat_func=stat_func,
                    mode=ci_mode,
                )
                low, up = abs(avg - low), abs(up - avg)
            else:
                avg, low, up = 0, 0, 0
            final_plot_df2["Absolute Error"].append(avg)
            final_plot_df2["Predicted Exclusion Score Range"].append(col_name)
            final_plot_df2["CI95_low"].append(low)
            final_plot_df2["CI95_up"].append(up)
        else:
            final_plot_df2["True Scores Proportion"].append(
                np.mean(plot_df2.iloc[:, i])
            )
    final_plot_df2 = pd.DataFrame(final_plot_df2)

    # Use a continuous colormap for the second plot based on True Scores Proportion
    cmap = get_cmap("coolwarm")
    norm = Normalize(
        vmin=final_plot_df2["True Scores Proportion"].min(),
        vmax=final_plot_df2["True Scores Proportion"].max(),
    )
    colors_plot2 = [cmap(norm(val)) for val in final_plot_df2["True Scores Proportion"]]

    # Plot 2: Vertical bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=final_plot_df2,
        x="Predicted Exclusion Score Range",
        y="Absolute Error",
        ax=ax,
        palette=colors_plot2,
    )
    intervals2 = np.array([final_plot_df2["CI95_low"], final_plot_df2["CI95_up"]])
    ax.errorbar(
        x=np.arange(final_plot_df2.shape[0]),
        y=final_plot_df2["Absolute Error"],
        yerr=intervals2,
        capsize=5,
        fmt="_",
        markersize=10,
        ecolor="black",
    )
    for i, patch in enumerate(ax.patches):
        x_center = patch.get_x() + patch.get_width() / 2.0
        y_top = patch.get_height() + intervals2[1, i]
        label = f"{patch.get_height():.3f}"
        ax.text(
            x_center,
            y_top + 0.003,
            label,
            ha="center",
            va="bottom",
            color="black",
            fontsize=12,
            fontweight="bold",
        )
    # Add a red horizontal line at y = 0.2 with a label
    ax.axhline(0.2, linestyle="--", color="red", linewidth=2, label="Threshold: 0.2")
    ax.legend(loc="upper right", fontsize=12)

    # Add a colorbar for the bar colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("True Scores Proportion", fontsize=14, fontweight="bold")

    ax.set_title(
        "Average Error vs. Predicted Exclusion Score Range",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted Exclusion Score Range", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Absolute Error", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path_bis = save_path + "_true_scores.pdf"
            plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def plot_err_by_org(path_df=None, ci_mode="bca", save_path=None, show=False):
    if path_df is None:
        results = pd.read_csv(
            "./Results/ablation_study/ho_None_LGBMRegressor_results.csv"
        )
    else:
        results = pd.read_csv(path_df)

    P_plot_df = results[results["Evaluation"].isin(ho_pathogen)]
    B_plot_df = results[results["Evaluation"].isin(ho_bacillus)]
    Int_plot_df = results[results["Evaluation"].isin(ho_interaction)]

    for df in [B_plot_df, P_plot_df]:
        ci_low = []
        ci_up = []
        for row in range(df.shape[0]):
            ho_name = df["Evaluation"].iloc[row]

            try:
                method_df = pd.read_csv("./Data/Datasets/fe_combinatoric_COI.csv")

                file_list = os.listdir("./Results/models/")
                file_list = ["./Results/models/" + file for file in file_list]
                model_file = [file for file in file_list if ho_name in file][0]

                with open(model_file, "rb") as f:
                    pipeline = pkl.load(f)

                X_train, X_test, _, y_true = retrieve_data(method_df, ho_name)

                X_test = pipeline[:-1].transform(X_test)  # preprocess X_test

                yhat = pipeline[-1].predict(X_test).reshape(-1, 1)
            except Exception as e:
                Warning("Using pickling failed, trying load_lgbm function..")
                pipeline, method_df = load_lgbm_model(
                    "./Results/models/",
                    "./Data/Datasets/fe_combinatoric_COI.csv",
                    ho_name,
                )
                X_train, X_test, _, y_true = retrieve_data(method_df, ho_name)
                pipeline[:-1].fit(X_train)
                X_test = pipeline[:-1].transform(X_test)

                yhat = pipeline[-1].predict(X_test).reshape(-1, 1)

            y_true = np.array(y_true).reshape(-1, 1)
            abs_err = np.abs(yhat - y_true)

            avg = np.mean(abs_err)
            low, up = compute_CI(
                abs_err,
                num_iter=5000,
                confidence=95,
                seed=62,
                stat_func=stat_func,
                mode=ci_mode,
            )
            low, up = abs(low - avg), abs(up - avg)
            ci_low.append(low)
            ci_up.append(up)
        df["CI95_low"] = ci_low
        df["CI95_up"] = ci_up

    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharey=True)

    # Unification des couleurs avec viridis
    cmap = sns.color_palette("viridis_r", as_cmap=True)

    # Pathogen MAE Bar Plot
    sns.barplot(
        P_plot_df,
        x="Evaluation",
        y="MAE",
        ax=ax[0],
        palette="viridis",
        edgecolor="black",
    )
    intervals = np.array([P_plot_df["CI95_low"], P_plot_df["CI95_up"]])
    ax[0].errorbar(
        x=np.arange(P_plot_df.shape[0]),
        y=P_plot_df["MAE"],
        yerr=intervals,
        fmt="o",
        capsize=5,
        color="black",
    )
    ax[0].set_xlabel("Pathogen", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Mean Absolute Error", fontsize=14, fontweight="bold")
    ax[0].tick_params(axis="x", rotation=45)
    # Add bar labels (in black) on top of each bar with ".3f" formatting.
    for patch in ax[0].patches:
        x_center = patch.get_x() + patch.get_width() / 2.0
        y_top = patch.get_height()
        label = f"{patch.get_height():.3f}"
        ax[0].text(
            x_center,
            y_top + 0.01,
            label,
            ha="center",
            va="bottom",
            color="black",
            fontsize=12,
            fontweight="bold",
        )
    ax[0].legend(loc="upper right", fontsize=12)

    # Bacillus MAE Bar Plot
    sns.barplot(
        B_plot_df,
        x="Evaluation",
        y="MAE",
        ax=ax[1],
        palette="viridis",
        edgecolor="black",
    )
    intervals = np.array([B_plot_df["CI95_low"], B_plot_df["CI95_up"]])
    ax[1].errorbar(
        x=np.arange(B_plot_df.shape[0]),
        y=B_plot_df["MAE"],
        yerr=intervals,
        fmt="o",
        capsize=5,
        color="black",
    )
    ax[1].set_xlabel("Bacillus", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("Mean Absolute Error", fontsize=14, fontweight="bold")
    ax[1].tick_params(axis="x", rotation=45)

    # Add bar labels (in white) at the center of each bar with ".3f" formatting.
    for patch in ax[1].patches:
        x_center = patch.get_x() + patch.get_width() / 2.0
        y_top = patch.get_height()
        label = f"{patch.get_height():.3f}"
        ax[1].text(
            x_center,
            y_top + 0.01,
            label,
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )
    ax[1].legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path_bis = save_path + "_orgs.pdf"
            plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    # Interaction Heatmap
    plt.figure(figsize=(6, 20))
    Int_plot_df.set_index("Evaluation", inplace=True)
    sns.heatmap(
        Int_plot_df[["MAE"]],
        annot=True,
        fmt=".3f",
        cmap="viridis_r",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Mean Absolute Error"},
        annot_kws={"size": 10},
        yticklabels=True,
    )
    plt.title("Interaction MAE Heatmap", fontsize=14, fontweight="bold")
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path_bis = save_path + "_int.pdf"
            plt.savefig(save_path_bis, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    # Summary of Worst & Best Predictions
    worst_int, worst_mae = Int_plot_df["MAE"].idxmax(), Int_plot_df["MAE"].max()
    best_int, best_mae = Int_plot_df["MAE"].idxmin(), Int_plot_df["MAE"].min()
    print(
        f"Worst Interaction: {worst_int}, MAE = {worst_mae:.3f} (n={Int_plot_df.loc[worst_int]['n_samples']})"
    )
    print(
        f"Best Interaction: {best_int}, MAE = {best_mae:.3f} (n={Int_plot_df.loc[best_int]['n_samples']})"
    )


def plot_global_SHAP(
    path_model_folder=None,
    path_df=None,
    ho_name="1234_x_S.en",
    save_path=None,
    show=False,
):
    try:
        if path_df is None:
            method_df = pd.read_csv("./Data/Datasets/fe_combinatoric_COI.csv")
        else:
            method_df = pd.read_csv(path_df)

        if path_model_folder is None:
            path_model_folder = "./Results/models/"

        file_list = os.listdir(path_model_folder)
        file_list = [path_model_folder + file for file in file_list]
        model_file = [file for file in file_list if ho_name in file][0]

        with open(model_file, "rb") as f:
            pipeline = pkl.load(f)

        X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)
        X_test = pipeline[:-1].transform(X_test)  # preprocess X_test

    except Exception as e:
        Warning("Using pickling failed, trying load_lgbm function..")
        pipeline, method_df = load_lgbm_model(path_model_folder, path_df, ho_name)
        X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)
        pipeline[:-1].fit(X_train)
        X_test = pipeline[:-1].transform(X_test)

    explainer = shap.TreeExplainer(pipeline[-1])
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 12))
    shap.plots.bar(explainer(X_test), show=False)
    plt.title("Feature Importances", fontsize=13, fontweight="bold")
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += f"_importances_{ho_name}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title("Feature value impact on model outputs", fontsize=13, fontweight="bold")

    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += f"_impact_{ho_name}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def plot_local_SHAP(
    path_model_folder=None,
    path_df=None,
    ho_name="1234_x_S.en",
    mode="worst",
    save_path=None,
    show=False,
):
    try:
        if path_df is None:
            method_df = pd.read_csv("./Data/Datasets/fe_combinatoric_COI.csv")
        else:
            method_df = pd.read_csv(path_df)

        if path_model_folder is None:
            path_model_folder = "./Results/models/"

        file_list = os.listdir(path_model_folder)
        file_list = [path_model_folder + file for file in file_list]
        model_file = [file for file in file_list if ho_name in file][0]

        with open(model_file, "rb") as f:
            pipeline = pkl.load(f)

        X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)

        X_test = pipeline[:-1].transform(X_test)  # preprocess X_test

        yhat = pipeline[-1].predict(X_test).reshape(-1, 1)
    except Exception as e:
        Warning("Using pickling failed, trying load_lgbm function..")
        pipeline, method_df = load_lgbm_model(path_model_folder, path_df, ho_name)

        X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)

        # Fit Preprocessing steps
        pipeline[:-1].fit(X_train)
        X_test = pipeline[:-1].transform(X_test)  # preprocess X_test

        yhat = pipeline[-1].predict(X_test).reshape(-1, 1)

    abs_err = np.abs(yhat - np.array(Y_test).reshape(-1, 1))
    idx = np.argmax(abs_err) if mode == "worst" else np.argmin(abs_err)

    explainer = shap.TreeExplainer(pipeline[-1])
    shap_values = explainer(X_test.iloc[idx : idx + 1, :])
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"SHAP Explanation of the {mode} prediction.")
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += f"_{ho_name}_{mode}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


# Import Alibi explainers
# from alibi.explainers import IntegratedGradients, Counterfactual


# def prepare_alibi(path_model_folder=None, path_df=None, ho_name="1234_x_S.en"):
#     """
#     Loads the LightGBM model and dataset, creates a preprocessing pipeline,
#     fits the pipeline on training data, transforms the test set, and returns:
#       - the pipeline,
#       - original test data (X_test, Y_test),
#       - a prediction function (predict_fn) that wraps the pipeline,
#       - the per-feature range (min, max) computed on the transformed test set.
#     """
#     from pipeline import create_pipeline

#     # Set default model folder if needed
#     if path_model_folder is None:
#         path_model_folder = "./Results/models/"
#     file_list = os.listdir(path_model_folder)
#     file_list = [os.path.join(path_model_folder, file) for file in file_list]
#     # Find the model file containing the given ho_name
#     model_file = [file for file in file_list if ho_name in file][0]

#     model = lgb.Booster(model_file=model_file)

#     # Load dataset
#     if path_df is None:
#         method_df = pd.read_csv("./Data/Datasets/combinatoric_COI.csv")
#     else:
#         method_df = pd.read_csv(path_df)

#     target = ["Score"]
#     cat_cols = ["Modele"]
#     remove_cols = ["Unnamed: 0", "B_sample_ID", "P_sample_ID", "Bacillus", "Pathogene"]
#     num_cols = [
#         col for col in method_df.columns if col not in cat_cols + remove_cols + target
#     ]

#     # Create pipeline (assumes your create_pipeline function returns a list-like pipeline
#     # where the last element is the estimator and the rest are preprocessing steps)
#     pipeline = create_pipeline(
#         num_cols,
#         cat_cols,
#         imputer="KNNImputer",
#         scaler="RobustScaler",
#         estimator=model,
#         model_name="LGBMRegressor",
#     )

#     # Retrieve train/test splits (assumes retrieve_data returns X_train, X_test, Y_train, Y_test)
#     X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)

#     # Fit the preprocessing pipeline (exclude estimator) and transform test set
#     pipeline[:-1].fit(X_train)
#     # Transform X_test; if returned as an array, convert it to a DataFrame
#     X_test_transf = pipeline[:-1].transform(X_test)
#     if not isinstance(X_test_transf, pd.DataFrame):
#         # Use column names from original training data
#         X_test_transf = pd.DataFrame(X_test_transf, columns=X_train.columns)

#     # Create a prediction function that applies the preprocessing before predicting.
#     def predict_fn(x):
#         # Expect x as a DataFrame; if not, try to convert it
#         if not isinstance(x, pd.DataFrame):
#             x = pd.DataFrame(x, columns=X_test_transf.columns)
#         # Ensure prediction output is 2D (n_samples, 1)
#         preds = pipeline[-1].predict(x)
#         return np.atleast_2d(preds).T

#     # Compute feature ranges on transformed test set for use in counterfactual search
#     feat_min = X_test_transf.min().values
#     feat_max = X_test_transf.max().values
#     feature_range = np.vstack([feat_min, feat_max])

#     return pipeline, X_test, Y_test, predict_fn, feature_range


# def plot_local_alibi(
#     path_model_folder=None,
#     path_df=None,
#     ho_name="1234_x_S.en",
#     mode="worst",
#     save_path=None,
#     show=False,
# ):
#     """
#     Generates a counterfactual for a given query instance using Alibi's Counterfactual
#     explainer and visualizes the local changes between the original instance and the counterfactual.

#     Parameters:
#       - path_model_folder, path_df, ho_name: parameters for data/model loading.
#       - mode (str): "worst" to select the instance with the highest prediction error,
#                     "best" to select the instance with the lowest error.
#       - save_path (str, optional): if provided, saves the figure.
#       - show (bool): whether to display the plot.
#     """
#     # Prepare data, prediction function, and feature range
#     pipeline, X_test, Y_test, predict_fn, feature_range = prepare_alibi(
#         path_model_folder, path_df, ho_name
#     )

#     # Transform X_test to get features used by the model
#     X_test_transf = pipeline[:-1].transform(X_test)
#     if not isinstance(X_test_transf, pd.DataFrame):
#         X_test_transf = pd.DataFrame(X_test_transf, columns=X_test.columns)

#     # Make predictions on X_test_transf
#     y_pred = predict_fn(X_test_transf).flatten()
#     y_true = np.array(Y_test).flatten()
#     abs_err = np.abs(y_pred - y_true)

#     # Select query instance based on error
#     idx = np.argmax(abs_err) if mode == "worst" else np.argmin(abs_err)
#     query_instance = X_test_transf.iloc[[idx]].copy()
#     target_value = y_true[idx]
#     print(f"Target value (true): {target_value}")
#     print("Query instance dtypes:\n", query_instance.dtypes)

#     # Set tolerance for counterfactual target proximity
#     tol = 0.05

#     # Initialize the Alibi Counterfactual explainer.
#     # Note: For a regression task, target is set to the desired output value.
#     cf_explainer = Counterfactual(
#         predict_fn,
#         shape=(1, X_test_transf.shape[1]),
#         target=target_value,
#         tol=tol,
#         feature_range=feature_range,
#         max_iter=1000,
#         lam_init=1e-1,
#         verbose=True,
#         early_stop=50,
#     )

#     # Generate counterfactual for the query instance
#     explanation = cf_explainer.explain(query_instance.to_numpy())

#     if explanation.cf is None:
#         raise ValueError("No counterfactual found.")

#     # Retrieve counterfactual and convert to DataFrame
#     cf_instance = pd.DataFrame(explanation.cf, columns=query_instance.columns)

#     # Create a comparison table: original, counterfactual, and absolute differences.
#     comp_df = query_instance.reset_index(drop=True).T.copy()
#     comp_df.columns = ["Original"]
#     comp_df["Counterfactual"] = cf_instance.T.iloc[:, 0]
#     comp_df["Absolute Difference"] = (
#         comp_df["Counterfactual"] - comp_df["Original"]
#     ).abs()

#     # Plot the table as a heatmap for differences.
#     plt.figure(figsize=(10, max(4, 0.4 * len(comp_df))))
#     # We display the absolute differences as a heatmap
#     ax = sns.heatmap(
#         comp_df[["Absolute Difference"]],
#         annot=True,
#         fmt=".3f",
#         cmap="RdBu_r",
#         cbar_kws={"label": "Absolute Difference"},
#         linewidths=0.5,
#         linecolor="black",
#         yticklabels=comp_df.index,
#     )
#     plt.title(
#         "Local Counterfactual Explanation (Alibi)", fontsize=16, fontweight="bold"
#     )
#     plt.ylabel("Features", fontsize=14, fontweight="bold")
#     plt.xlabel("")
#     plt.tight_layout()

#     if save_path is not None:
#         if not save_path.endswith(".pdf"):
#             save_path += f"_{ho_name}_{mode}.pdf"
#         plt.savefig(save_path, format="pdf", bbox_inches="tight")
#     if show:
#         plt.show()

#     # Also print the comparison table
#     print("Comparison of Original and Counterfactual:")
#     print(comp_df)

#     return explanation, comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis plots.")
    parser.add_argument(
        "plot_type",
        choices=[
            "show_perf_skewedness",
            "plot_model_selection",
            "summary_model_selection",
            "summary_preprocess_selection",
            "plot_native_feature_selection",
            "plot_feature_engineering",
            "plot_optuna_study",
            "plot_feature_importance_heatmap",
            "plot_ablation_study",
            "plot_err_distrib",
            "plot_err_by_org",
            "plot_global_SHAP",
            "plot_local_SHAP",
            "plot_global_DiCE",
            "plot_local_DiCE",
        ],
        help="Type of plot to generate.",
    )
    args = parser.parse_args()

    plot_type = args.plot_type

    if plot_type == "show_perf_skewedness":
        print("Running show_perf_skewedness and saving to console.")
        show_perf_skewedness(
            "LGBMRegressor", path_df="./Results/model_selection/ho_all_results.csv"
        )
    elif plot_type == "plot_model_selection":
        print(
            "Running plot_model_selection for MAE and saving to ./Plots/model_selection_MAE.pdf"
        )
        plot_model_selection(
            "./Results/model_selection/ho_all_results.csv",
            "MAE",
            mode="all",
            avg_mode="weighted",
            save_path="./Plots/model_selection_MAE.pdf",
            show=False,
        )
        print(
            "Running plot_model_selection for RMSE and saving to ./Plots/model_selection_RMSE.pdf"
        )
        plot_model_selection(
            "./Results/model_selection/ho_all_results.csv",
            "RMSE",
            mode="all",
            avg_mode="weighted",
            save_path="./Plots/model_selection_RMSE.pdf",
            show=False,
        )
    elif plot_type == "summary_model_selection":
        print(
            "Running summary_model_selection for RMSE and saving to ./Plots/summary_model_selection_RMSE.pdf"
        )
        summary_model_selection(
            "./Results/model_selection/ho_all_results.csv",
            metric="RMSE",
            method="combinatoric",
            avg_mode="weighted",
            save_path="./Plots/summary_model_selection_RMSE.pdf",
            show=False,
        )
        print(
            "Running summary_model_selection for MAE and saving to ./Plots/summary_model_selection_MAE.pdf"
        )
        summary_model_selection(
            "./Results/model_selection/ho_all_results.csv",
            metric="MAE",
            method="combinatoric",
            avg_mode="weighted",
            save_path="./Plots/summary_model_selection_MAE.pdf",
            show=False,
        )
    elif plot_type == "summary_preprocess_selection":
        print(
            "Running summary_preprocess_selection for MAE and saving to ./Plots/preprocess_selection_MAE.pdf"
        )
        summary_preprocess_selection(
            "./Results/preprocess_selection/ho_all_results.csv",
            metric="MAE",
            method="combinatoric",
            avg_mode="weighted",
            save_path="./Plots/preprocess_selection_MAE.pdf",
            show=False,
        )

        print(
            "Running summary_preprocess_selection for RMSE and saving to ./Plots/preprocess_selection_MAE.pdf"
        )
        summary_preprocess_selection(
            "./Results/preprocess_selection/ho_all_results.csv",
            metric="RMSE",
            method="combinatoric",
            avg_mode="weighted",
            save_path="./Plots/preprocess_selection_MAE.pdf",
            show=False,
        )
    elif plot_type == "plot_native_feature_selection":
        print(
            "Running plot_native_feature_selection and saving to ./Plots/native_feature_selection.pdf"
        )
        plot_native_feature_selection(
            "./Results/native_feature_selection/",  # step_1_LGBMRegressor_controled_homology_permutation_details.csv",
            ci_mode="bca",
            save_path="./Plots/native_feature_selection.pdf",
            show=False,
        )
    elif plot_type == "plot_feature_engineering":
        print(
            "Running plot_feature_engineering and saving to ./Plots/feature_engineering.pdf"
        )
        plot_feature_engineering(
            "./Results/feature_engineering/",  # step_1_LGBMRegressor_controled_homology_permutation_details.csv",
            ci_mode="bca",
            save_path="./Plots/feature_engineering",
            show=False,
        )
    elif plot_type == "plot_optuna_study":
        print("Running plot_optuna_study and saving to ./Plots/optuna_study.pdf")
        plot_optuna_study(
            "./Results/optuna_campaign/optuna_study.pkl",
            save_path="./Plots/optuna_study",
            show=False,
        )
    elif plot_type == "plot_feature_importance_heatmap":
        print(
            "Running plot_feature_importance_heatmap and saving to ./Plots/feature_importances_GAIN.pdf"
        )
        plot_feature_importance_heatmap(
            "./Results/models/",
            save_path="./Plots/feature_importances_GAIN.pdf",
            show=False,
        )
    elif plot_type == "plot_ablation_study":
        print("Running plot_ablation_study and saving to ./Plots/ablation_study.pdf")
        plot_ablation_study(
            "./Results/ablation_study/",
            save_path="./Plots/ablation_study",
            show=False,
        )
    elif plot_type == "plot_err_distrib":
        print("Running plot_err_distrib and saving to ./Plots/distrib_err.pdf")
        plot_err_distrib(
            "./Results/ablation_study/ho_None_LGBMRegressor_results.csv",
            save_path="./Plots/distrib_err",
            show=False,
        )
    elif plot_type == "plot_err_by_org":
        print("Running plot_err_by_org and saving to ./Plots/err_by_org.pdf")
        plot_err_by_org(
            "./Results/ablation_study/ho_None_LGBMRegressor_results.csv",
            save_path="./Plots/err_by_org",
            show=False,
        )

    elif plot_type == "plot_global_SHAP":
        print("Running plot_global_SHAP and saving to ./Plots/global_SHAP.pdf")
        plot_global_SHAP(
            ho_name="1234_x_S.en", save_path="./Plots/global_SHAP", show=False
        )
    elif plot_type == "plot_local_SHAP":
        print(
            "Running plot_local_SHAP (worst) and saving to ./Plots/local_SHAP_worst.pdf"
        )
        plot_local_SHAP(
            ho_name="1234_x_S.en",
            mode="worst",
            save_path="./Plots/local_SHAP",
            show=False,
        )
        print(
            "Running plot_local_SHAP (best) and saving to ./Plots/local_SHAP_best.pdf"
        )
        plot_local_SHAP(
            ho_name="1234_x_S.en",
            mode="best",
            save_path="./Plots/local_SHAP",
            show=False,
        )
    # elif plot_type == "plot_global_DiCE":
    #     print("Running plot_global_DiCE and saving to ./Plots/global_DiCE.pdf")
    #     plot_global_DiCE(
    #         ho_name="1234_x_S.en", save_path="./Plots/global_DiCE", show=False
    #     )
    # elif plot_type == "plot_local_DiCE":
    #     print(
    #         "Running plot_local_DiCE (worst) and saving to ./Plots/local_DiCE_worst.pdf"
    #     )
    #     plot_local_DiCE(
    #         ho_name="1234_x_S.en",
    #         mode="worst",
    #         save_path="./Plots/local_DiCE",
    #         show=False,
    #     )
    #     print(
    #         "Running plot_local_DiCE (best) and saving to ./Plots/local_DiCE_best.pdf"
    #     )
    #     plot_local_DiCE(
    #         ho_name="1234_x_S.en",
    #         mode="best",
    #         save_path="./Plots/local_DiCE",
    #         show=False,
    #     )
