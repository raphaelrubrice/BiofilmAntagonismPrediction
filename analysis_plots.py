import pandas as pd
import numpy as np
import pickle as pkl
import os, re, argparse

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
        ci_data = (
            df.groupby("Model")
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
        ci_data = (
            df.groupby("Model")
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
    Plot the Permutation Feature Importance (PFI) with 95% Confidence Intervals.

    Parameters:
    - path_df (str, optional): Path to CSV file containing feature selection results.
    - ci_mode (str): Mode for computing confidence intervals (default: "t").

    This function reads the results, calculates mean PFI values, and visualizes
    them as a bar plot with error bars representing confidence intervals.
    """

    # Load results
    if path_df is None:
        results = pd.read_csv(
            "./Results/native_feature_selection/step_1_LGBMRegressor_controled_homology_permutation_details.csv"
        )
    else:
        results = pd.read_csv(path_df)

    # Initialize dictionary to store feature importance data
    plot_df = {"Feature": [], "PFI": [], "CI95_low": [], "CI95_up": []}

    # Compute PFI and Confidence Intervals for each feature permutation
    for permutation in pd.unique(results["Permutation"]):
        if permutation != "No Permutation":
            mask = results["Permutation"] == permutation
            avg = results[mask]["diff_RMSE"].mean()

            if avg != 0:
                low, up = compute_CI(
                    results[mask]["diff_RMSE"],
                    num_iter=5000,
                    confidence=95,
                    seed=62,
                    mode=ci_mode,
                )
                # Specify the size not the absolute values of the CI
                low, up = abs(avg - low), abs(up - avg)
            else:
                avg, low, up = 0, 0, 0

            plot_df["Feature"].append(permutation)
            plot_df["PFI"].append(avg)
            plot_df["CI95_low"].append(low)
            plot_df["CI95_up"].append(up)

    # Convert to DataFrame and sort by importance
    plot_df = pd.DataFrame(plot_df).sort_values("PFI", ascending=False)

    # Set seaborn style for a clean look
    sns.set_theme(style="whitegrid")

    # Define figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot
    sns.barplot(
        data=plot_df,
        y="Feature",
        x="PFI",
        orient="h",
        hue="Feature",
        palette="magma_r",  # High-contrast perceptually uniform colormap
        edgecolor="black",
    )

    # Confidence Interval error bars
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

    # Improve labels and title
    ax.set_xlabel(
        "Permutation Feature Importance (PFI)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Feature", fontsize=14, fontweight="bold")
    ax.set_title(
        "Feature Importances with 95% Confidence Intervals",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust layout
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
    Plots feature engineering results, including Permutation Feature Importance (PFI)
    with confidence intervals, and a heatmap of RMSE, MAE, and PFI.

    Parameters:
    - path_df (str, optional): Path to CSV file containing feature engineering results.
    - ci_mode (str): Mode for computing confidence intervals (default: "t").

    The function generates:
    1. A bar plot for the top 10 features sorted by PFI, with error bars for CI95.
    2. A heatmap displaying RMSE, MAE, and PFI across all features.
    """

    # Load results
    if path_df is None:
        results = pd.read_csv(
            "./Results/feature_engineering/step_1_LGBMRegressor_controled_homology_permutation_details.csv"
        )
    else:
        results = pd.read_csv(path_df)

    # Initialize dictionary to store feature importance data
    plot_df = {
        "Feature": [],
        "RMSE": [],
        "MAE": [],
        "PFI": [],
        "CI95_low": [],
        "CI95_up": [],
    }

    # Compute metrics and Confidence Intervals for each feature permutation
    for permutation in pd.unique(results["Permutation"]):
        if permutation != "No Permutation":
            mask = results["Permutation"] == permutation
            rmse = results[mask]["RMSE"].mean()
            mae = results[mask]["MAE"].mean()
            avg = results[mask]["diff_RMSE"].mean()

            if avg != 0:
                low, up = compute_CI(
                    results[mask]["diff_RMSE"],
                    num_iter=5000,
                    confidence=95,
                    seed=62,
                    mode=ci_mode,
                )
                # Specify the size not the absolute values of the CI
                low, up = abs(avg - low), abs(up - avg)
            else:
                avg, low, up = 0, 0, 0

            plot_df["Feature"].append(permutation)
            plot_df["RMSE"].append(rmse)
            plot_df["MAE"].append(mae)
            plot_df["PFI"].append(avg)
            plot_df["CI95_low"].append(low)
            plot_df["CI95_up"].append(up)

    # Convert to DataFrame and sort by PFI
    plot_df = pd.DataFrame(plot_df).sort_values("PFI", ascending=False)

    # Select top 10 features
    top_features = plot_df.iloc[:10]

    # Set seaborn theme for clean visualization
    sns.set_theme(style="whitegrid")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_features,
        y="Feature",
        x="PFI",
        orient="h",
        palette="magma_r",  # High-contrast perceptually uniform colormap
        edgecolor="black",
    )

    # Confidence Interval error bars
    intervals = np.array([top_features["CI95_low"], top_features["CI95_up"]])
    ax.errorbar(
        y=np.arange(top_features.shape[0]),
        x=top_features["PFI"],
        xerr=intervals,
        fmt="o",
        capsize=4,
        elinewidth=1.5,
        color="black",
        alpha=0.8,
    )

    # Improve labels and title
    ax.set_xlabel(
        "Permutation Feature Importance (PFI)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Feature", fontsize=14, fontweight="bold")
    ax.set_title(
        "Top 10 Feature Importances with Confidence Intervals",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    # Set feature names as index for heatmap
    plot_df.set_index("Feature", inplace=True)

    # Create heatmap
    plt.figure(figsize=(12, 26))
    sns.heatmap(
        plot_df[["PFI", "CI95_low", "CI95_up"]],
        cmap="mako",  # Better contrast for metric comparison
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
    )

    # Heatmap title
    plt.title("Feature Engineering Metrics Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


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

    # Plot parameter ranking
    plot_rank(
        optuna_study,
        params=[
            "n_estimatorsnum_leaves",
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
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
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
        fmt=".2f",
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
    Black contours are added for each bar.

    Parameters:
    - path_ablation_folder (str, optional): Path to the folder containing ablation results.
    - weighted (bool): Whether to weight RMSE/MAE by the number of samples.
    - ci_mode (str): Confidence interval calculation method.
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
        df = pd.read_csv(os.path.join(path_ablation_folder, ablation))
        # Compute weighted or mean RMSE & MAE
        if weighted:
            # Note: here we are not normalizing by total samples so that differences remain absolute.
            df["RMSE"] *= df["n_samples"]
            df["MAE"] *= df["n_samples"]
            rmse, mae = df["RMSE"].sum(), df["MAE"].sum()
        else:
            rmse, mae = df["RMSE"].mean(), df["MAE"].mean()

        plot_df["Features"].append(keys[i])
        plot_df[addon + "MAE"].append(mae)
        plot_df[addon + "RMSE"].append(rmse)

        # Compute confidence intervals
        func = weighted_stat_func if weighted else np.mean
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

    # For each metric, create bar plots with custom colors
    sns.set_theme(style="whitegrid")
    for metric in ["RMSE", "MAE"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Reorder so that baseline ("None removed") is first
        baseline_df = plot_df[plot_df["Features"] == "None removed"]
        others = plot_df[plot_df["Features"] != "None removed"].sort_values(
            by=addon + metric, ascending=False
        )
        df_plot = pd.concat([baseline_df, others], axis=0).reset_index(drop=True)
        print(df_plot)
        # Get the baseline metric value (assumed to be the first row)
        baseline_value = df_plot.iloc[0][addon + metric]

        # Compute maximum differences for normalization of increases and decreases
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

        # Build a list of colors based on the difference relative to baseline:
        # Baseline gets gray; if value > baseline, use Reds; if < baseline, use Greens.
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

        # Draw the barplot with the custom palette (here, we pass the colors list)
        barplot = sns.barplot(
            data=df_plot,
            y="Features",
            x=addon + metric,
            orient="h",
            ax=ax,
            palette=colors,
        )

        # Add error bars with confidence intervals
        intervals = np.array(
            [df_plot[f"{metric}_CI95_low"], df_plot[f"{metric}_CI95_up"]]
        )
        ax.errorbar(
            x=df_plot[addon + metric],
            y=np.arange(len(df_plot)),
            xerr=intervals,
            capsize=5,
            fmt="o",
            ecolor="black",
        )

        # Add black contours to each bar
        for patch in ax.patches:
            patch.set_edgecolor("black")
            patch.set_linewidth(1)

        # Set titles and labels
        ax.set_title(
            f"Ablation Study: Impact on {addon + metric}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"{addon + metric} Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ablated Feature", fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path is not None:
            if not save_path.endswith(".pdf"):
                save_path += ".pdf"
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
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
        method_df = pd.read_csv("./Data/Datasets/combinatoric_COI.csv")
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
        pipeline, method_df = load_lgbm_model(
            "./Results/models/", "./Data/Datasets/combinatoric_COI.csv", ho_name
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
    ax.set_title("Distribution of Prediction Errors", fontsize=16, fontweight="bold")
    ax.set_xlabel("Error Threshold", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage of Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
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
            np.mean(abs_err[yhat < 0.2])
            if not check_empty(abs_err, yhat < 0.2)
            else np.nan
        )
        n_0_2 = 0 if np.isnan(_0_2) else np.mean(yhat < 0.2)
        _0_4 = (
            np.mean(abs_err[(yhat >= 0.2) & (yhat < 0.4)])
            if not check_empty(abs_err, (yhat >= 0.2) & (yhat < 0.4))
            else np.nan
        )
        n_0_4 = 0 if np.isnan(_0_4) else np.mean((yhat >= 0.2) & (yhat < 0.4))
        _0_6 = (
            np.mean(abs_err[(yhat >= 0.4) & (yhat < 0.6)])
            if not check_empty(abs_err, (yhat >= 0.4) & (yhat < 0.6))
            else np.nan
        )
        n_0_6 = 0 if np.isnan(_0_6) else np.mean((yhat >= 0.4) & (yhat < 0.6))
        _0_8 = (
            np.mean(abs_err[(yhat >= 0.6) & (yhat < 0.8)])
            if not check_empty(abs_err, (yhat >= 0.6) & (yhat < 0.8))
            else np.nan
        )
        n_0_8 = 0 if np.isnan(_0_8) else np.mean((yhat >= 0.6) & (yhat < 0.8))
        sup_0_8 = (
            np.mean(abs_err[yhat >= 0.8])
            if not check_empty(abs_err, yhat >= 0.8)
            else np.nan
        )
        n_sup_0_8 = 0 if np.isnan(sup_0_8) else np.mean(yhat >= 0.8)

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
            avg = np.mean(plot_df2.iloc[:, i])
            low, up = compute_CI(
                plot_df2.iloc[:, i],
                num_iter=5000,
                confidence=95,
                seed=62,
                stat_func=stat_func,
                mode=ci_mode,
            )
            low, up = abs(avg - low), abs(up - avg)
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
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
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

            pipeline, method_df = load_lgbm_model(
                "./Results/models/", "./Data/Datasets/combinatoric_COI.csv", ho_name
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

    fig, ax = plt.subplots(2, 1, figsize=(8, 12), sharey=True)

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

    plt.tight_layout()
    plt.show()

    # Interaction Heatmap
    plt.figure(figsize=(6, 20))
    Int_plot_df.set_index("Evaluation", inplace=True)
    sns.heatmap(
        Int_plot_df[["MAE"]],
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Mean Absolute Error"},
        annot_kws={"size": 12},
    )
    plt.title("Interaction MAE Heatmap", fontsize=14, fontweight="bold")
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    # Summary of Worst & Best Predictions
    worst_int, worst_mae = Int_plot_df["MAE"].idxmax(), Int_plot_df["MAE"].max()
    best_int, best_mae = Int_plot_df["MAE"].idxmin(), Int_plot_df["MAE"].min()
    print(
        f"Worst Interaction: {worst_int}, MAE = {worst_mae:.2f} (n={Int_plot_df.loc[worst_int]['n_samples']})"
    )
    print(
        f"Best Interaction: {best_int}, MAE = {best_mae:.2f} (n={Int_plot_df.loc[best_int]['n_samples']})"
    )


def plot_global_SHAP(
    path_model_folder=None,
    path_df=None,
    ho_name="1234_x_S.en",
    save_path=None,
    show=False,
):
    pipeline, method_df = load_lgbm_model(path_model_folder, path_df, ho_name)

    X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)

    # Fit Preprocessing steps
    pipeline[:-1].fit(X_train)
    X_test = pipeline[:-1].transform(X_test)  # preprocess X_test

    explainer = shap.TreeExplainer(pipeline[-1])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
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
    shap.plots.waterfall(shap_values[0])
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def prepare_dice(path_model_folder=None, path_df=None, ho_name="1234_x_S.en"):
    from pipeline import create_pipeline

    if path_model_folder is None:
        path_model_folder = "./Results/models/"
    file_list = os.listdir(path_model_folder)
    file_list = [path_model_folder + file for file in file_list]
    model_file = [file for file in file_list if ho_name in file][0]

    model = lgb.Booster(model_file=model_file)

    if path_df is None:
        method_df = pd.read_csv("./Data/Datasets/combinatoric_COI.csv")
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

    pipeline = create_pipeline(
        num_cols,
        cat_cols,
        imputer="KNNImputer",
        scaler="RobustScaler",
        estimator=model,
        model_name="LGBMRegressor",
    )

    X_train, X_test, Y_train, Y_test = retrieve_data(method_df, ho_name)

    X_train, X_test, _, y_true = retrieve_data(method_df, ho_name)
    pipeline[:-1].fit(X_train)
    X_test_transf = pipeline[:-1].transform(X_test)

    # Make sure we convert categorical features to suitable format for DiCE
    combined_df = pd.concat([X_test_transf, Y_test.reset_index(drop=True)], axis=1)

    # Ensure all model columns are float
    continuous_features = num_cols.copy()
    if "Modele_I" in combined_df.columns:
        continuous_features.extend(["Modele_I", "Modele_II", "Modele_III"])
        combined_df["Modele_I"] = combined_df["Modele_I"].astype(float)
        combined_df["Modele_II"] = combined_df["Modele_II"].astype(float)
        combined_df["Modele_III"] = combined_df["Modele_III"].astype(float)

    dice_data = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=continuous_features,
        outcome_name="Score",
    )

    dice_model = dice_ml.Model(model=model, model_type="regressor", backend="sklearn")
    exp = dice_ml.Dice(dice_data, dice_model, method="genetic")

    return pipeline, X_test, Y_test, dice_data, dice_model, exp


def plot_global_DiCE(
    path_model_folder=None,
    path_df=None,
    ho_name="1234_x_S.en",
    save_path=None,
    show=False,
):
    """
    Plots global feature importances computed using DiCE with a genetic algorithm.
    Assumes that a DiCE explainer (exp) has been constructed.
    """
    _, _, _, data, model, exp = prepare_dice(path_model_folder, path_df, ho_name)

    # Get the range of target values in the dataset
    y_min = data.dataframe["Score"].min()
    y_max = data.dataframe["Score"].max()

    # For regression tasks, we need to specify a desired range
    # Using a range that covers the middle 60% of the target distribution
    target_range = [y_max - 0.2, y_max]

    # Compute global feature importances with desired_range parameter
    global_imp = exp.global_feature_importance(data, desired_range=target_range)

    # Plot as a publication-grade bar plot
    ax = global_imp.plot(
        kind="bar", figsize=(10, 6), color="steelblue", edgecolor="black"
    )
    ax.set_title(
        "Global Feature Importances (DiCE Genetic)", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Features", fontsize=14, fontweight="bold")
    ax.set_ylabel("Importance", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path is not None:
        if not save_path.endswith(".pdf"):
            save_path += ".pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def plot_local_DiCE(
    path_model_folder=None,
    path_df=None,
    ho_name="1234_x_S.en",
    mode="worst",
    save_path=None,
    show=False,
):
    """
    Generates 5 counterfactuals for a given query instance and visualizes the local feature importances
    computed using DiCE with a genetic algorithm. Instead of a barplot, a heatmap of the local feature
    importances is produced for a publication-grade figure.

    Parameters:
    - path_model_folder (str, optional): Path to the folder containing models.
    - path_df (str, optional): Path to the dataset.
    - ho_name (str): Evaluation identifier.
    - mode (str): "worst" to select instance with minimum prediction error; "best" for maximum.
    """
    # Prepare the DiCE objects and retrieve data/model/explainer
    pipeline, X_test, Y_test, dice_data, model, exp = prepare_dice(
        path_model_folder, path_df, ho_name
    )

    # Transform X_test using the pipeline (excluding the estimator)
    X_test_t = pipeline[:-1].transform(X_test)

    # If transformation returns an array, convert it to DataFrame using dice_data.dataframe column names (excluding the outcome)
    if not isinstance(X_test_t, pd.DataFrame):
        feature_cols = [col for col in dice_data.dataframe.columns if col != "Score"]
        X_test_t = pd.DataFrame(X_test_t, columns=feature_cols)

    # Make predictions
    yhat = pipeline[-1].predict(X_test_t).reshape(-1, 1)
    y_true = np.array(Y_test).reshape(-1, 1)
    abs_err = np.abs(yhat - y_true)

    # Select query instance: choose worst (highest error) or best (lowest error)
    # Note: Changed logic - "worst" should be highest error, "best" should be lowest error
    idx = np.argmax(abs_err) if mode == "worst" else np.argmin(abs_err)

    # Select the query instance and ensure all values are float
    query_instance = X_test_t.iloc[[idx]].copy()
    # Convert all columns to float
    for col in query_instance.columns:
        query_instance[col] = query_instance[col].astype(float)

    target = y_true[idx]
    print(f"Target value: {target}")
    print("Query instance dtypes:\n", query_instance.dtypes)

    # Define desired range for counterfactuals (a small range around the target)
    desired_range = [target - 0.05, target + 0.05]

    try:
        # Generate 5 counterfactuals for the query instance using random initialization
        # This avoids KD-tree issues
        cf = exp.generate_counterfactuals(
            query_instance,
            total_CFs=5,
            desired_range=desired_range,
            initialization="random",
        )

        # Compute local feature importances
        local_imp = exp.local_feature_importance(cf)

        # Ensure we have a DataFrame with one column named "Importance"
        if not isinstance(local_imp, pd.DataFrame):
            local_imp = pd.DataFrame(local_imp, columns=["Importance"])

        # Sort by importance for better visualization
        local_imp = local_imp.sort_values(by="Importance", ascending=False)

        # Plot a heatmap of the local feature importances
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            local_imp.values.reshape(-1, 1),
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Local Importance"},
            linewidths=0.5,
            linecolor="black",
            yticklabels=local_imp.index,
        )
        ax.set_title(
            "Local Feature Importances (DiCE Genetic)", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("", fontsize=14)
        ax.set_ylabel("Features", fontsize=14, fontweight="bold")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return cf, local_imp

    except Exception as e:
        print(f"Error generating counterfactuals: {str(e)}")
        print("Trying alternative approach...")

        # Alternative approach: Generate random counterfactuals
        # This is a fallback if the genetic method fails
        from sklearn.neighbors import NearestNeighbors

        # Get similar instances from the dataset
        X_sample = dice_data.dataframe.drop(columns=["Score"]).sample(
            min(100, len(dice_data.dataframe))
        )
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X_sample)
        _, indices = nn.kneighbors(query_instance)

        print("Using nearest neighbors as counterfactuals")

        # Create a simple visualization of feature differences
        neighbors = X_sample.iloc[indices[0]]
        diff_df = neighbors - query_instance.values

        # Plot the differences
        plt.figure(figsize=(12, 8))
        sns.heatmap(diff_df.transpose(), cmap="RdBu_r", center=0, annot=True, fmt=".2f")
        plt.title(
            "Feature Differences in Similar Instances (Alternative to DiCE)",
            fontsize=16,
        )
        plt.ylabel("Features", fontsize=14)
        plt.xlabel("Similar Instance", fontsize=14)
        plt.tight_layout()
        if save_path is not None:
            if not save_path.endswith(".pdf"):
                save_path += ".pdf"
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        if show:
            plt.show()

        return neighbors, diff_df.abs().mean().sort_values(ascending=False)


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
            "./Results/native_feature_selection/step_1_LGBMRegressor_controled_homology_permutation_details.csv",
            ci_mode="bca",
            save_path="./Plots/native_feature_selection.pdf",
            show=False,
        )
    elif plot_type == "plot_feature_engineering":
        print(
            "Running plot_feature_engineering and saving to ./Plots/feature_engineering.pdf"
        )
        plot_feature_engineering(
            "./Results/feature_engineering/step_1_LGBMRegressor_controled_homology_permutation_details.csv",
            ci_mode="bca",
            save_path="./Plots/feature_engineering.pdf",
            show=False,
        )
    elif plot_type == "plot_optuna_study":
        print("Running plot_optuna_study and saving to ./Plots/optuna_study.pdf")
        plot_optuna_study(
            "./Results/optuna_campaign/optuna_study.pkl",
            save_path="./Plots/optuna_study.pdf",
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
            save_path="./Plots/ablation_study.pdf",
            show=False,
        )
    elif plot_type == "plot_err_distrib":
        print("Running plot_err_distrib and saving to ./Plots/distrib_err.pdf")
        plot_err_distrib(
            "./Results/ablation_study/ho_None_LGBMRegressor_results.csv",
            save_path="./Plots/distrib_err.pdf",
            show=False,
        )
    elif plot_type == "plot_err_by_org":
        print("Running plot_err_by_org and saving to ./Plots/err_by_org.pdf")
        plot_err_by_org(
            "./Results/ablation_study/ho_None_LGBMRegressor_results.csv",
            save_path="./Plots/err_by_org.pdf",
            show=False,
        )

    elif plot_type == "plot_global_SHAP":
        print("Running plot_global_SHAP and saving to ./Plots/global_SHAP.pdf")
        plot_global_SHAP(
            ho_name="1234_x_S.en", save_path="./Plots/global_SHAP.pdf", show=False
        )
    elif plot_type == "plot_local_SHAP":
        print(
            "Running plot_local_SHAP (worst) and saving to ./Plots/local_SHAP_worst.pdf"
        )
        plot_local_SHAP(
            ho_name="1234_x_S.en",
            mode="worst",
            save_path="./Plots/local_SHAP_worst.pdf",
            show=False,
        )
        print(
            "Running plot_local_SHAP (best) and saving to ./Plots/local_SHAP_best.pdf"
        )
        plot_local_SHAP(
            ho_name="1234_x_S.en",
            mode="best",
            save_path="./Plots/local_SHAP_best.pdf",
            show=False,
        )
    elif plot_type == "plot_global_DiCE":
        print("Running plot_global_DiCE and saving to ./Plots/global_DiCE.pdf")
        plot_global_DiCE(
            ho_name="1234_x_S.en", save_path="./Plots/global_DiCE.pdf", show=False
        )
    elif plot_type == "plot_local_DiCE":
        print(
            "Running plot_local_DiCE (worst) and saving to ./Plots/local_DiCE_worst.pdf"
        )
        plot_local_DiCE(
            ho_name="1234_x_S.en",
            mode="worst",
            save_path="./Plots/local_DiCE_worst.pdf",
            show=False,
        )
        print(
            "Running plot_local_DiCE (best) and saving to ./Plots/local_DiCE_best.pdf"
        )
        plot_local_DiCE(
            ho_name="1234_x_S.en",
            mode="best",
            save_path="./Plots/local_DiCE_best.pdf",
            show=False,
        )
