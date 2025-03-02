import os
import json
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from datasets import all_possible_hold_outs

all_ho_names = all_possible_hold_outs(return_names=True)
ho_org = [name for name in all_ho_names if "_x_" not in name]
ho_interaction = [name for name in all_ho_names if name not in ho_org]
ho_pathogen = ["E.ce", "E.co", "S.au", "S.en"]
ho_bacillus = [name for name in ho_org if name not in ho_pathogen]

std_keys_dict = {"mae": "std_abs_err", "mape": "std_relative_abs_err"}


def get_cv_results(file, metric):
    assert "cv" in file, "Must be a cv result format"
    with open(file, "r") as f:
        results = json.load(f)

    METHODS = ["avg", "random", "combinatoric"]
    PERFS = []
    STDS = []
    for method in METHODS:
        perfs = []
        if metric in ["mae", "mape"]:
            std = []
        else:
            std = None
        for fold in results[method].keys():
            perfs.append(results[method][fold][metric])
            if std is not None:
                std.append(results[method][fold][std_keys_dict[metric]])
        PERFS.append(np.mean(perfs))
        if std is not None:
            STDS.append(np.mean(std))
        else:
            STDS.append(np.nan)
    return pd.DataFrame({"Dataset": METHODS, metric.upper(): PERFS, "Std": STDS})


# def plot_model_selection(results_file_path, metric='RMSE'):
#     # Validate metric choice
#     assert metric in ["RMSE", "MAE", "std_abs_err", "MAPE", "std_abs_relative_err", "R2"], \
#         "metric must be one of: RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2"

#     # Read results and compute mean performance per Method and Model
#     results = pd.read_csv(results_file_path)
#     perfs = results.groupby(["Method", "Model"])[metric].mean().reset_index()
#     perfs = perfs.sort_values(metric)

#     # If metric is MAE or MAPE, also compute error bars using the corresponding error column
#     if metric in ["MAE", "MAPE"]:
#         # Determine error column based on metric
#         err_metric = "std_abs_err" if metric == "MAE" else "std_abs_relative_err"
#         # Aggregate both the metric and its associated error using mean
#         agg_data = results.groupby(["Method", "Model"]).agg({metric: 'mean', err_metric: 'mean'}).reset_index()
#     else:
#         agg_data = results.groupby(["Method", "Model"])[metric].mean().reset_index()

#     # Sort by metric value (lower is better)
#     agg_data = agg_data.sort_values(metric)

#     # Set a publication style
#     sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1)
#     plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif', "font.weight": 'bold'})

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Draw barplot with seaborn
#     ax = sns.barplot(data=perfs, x="Model", y=metric, hue="Method", palette="inferno", ax=ax,
#                      order=["LinearRegression", "Ridge", "Lasso", "ElasticNet", "LinearSVR",
#                             "RandomForestRegressor", "GradientBoostingRegressor", "LGBMRegressor", "XGBRegressor"])

#     # Add a horizontal dotted line at the overall lowest metric value (assuming lower is better)
#     if metric == "R2":
#         best_value = perfs[metric].max()
#     else:
#         best_value = perfs[metric].min()
#     ax.axhline(y=best_value, linestyle=":", color="black", linewidth=2, label=f"overall best {best_value:.3f}")

#     # Add a descriptive title and update axis labels with improved formatting.
#     ax.set_title(f"Model Selection Performance ({metric})", fontsize=16, fontweight='bold')
#     ax.set_xlabel("Model", fontsize=13, fontweight='bold')
#     ax.set_ylabel(metric, fontsize=13, fontweight='bold')

#     # Improve legend formatting.
#     legend = ax.legend(title="Method", title_fontsize=12, fontsize=10)
#     legend.get_frame().set_edgecolor('black')

#     plt.xticks(rotation=15)
#     plt.tight_layout()
#     plt.show()


def plot_model_selection(
    results_file_path, metric="RMSE", mode="all", save_path=None, show=False
):
    # Validate metric choice
    assert metric in [
        "RMSE",
        "MAE",
        "std_abs_err",
        "MAPE",
        "std_abs_relative_err",
        "R2",
    ], "metric must be one of: RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2"

    # Read results from CSV
    results = pd.read_csv(results_file_path)

    if mode == "bacillus":
        results = results[results["Evaluation"].isin(ho_bacillus)]
    elif mode == "pathogen":
        results = results[results["Evaluation"].isin(ho_pathogen)]
    elif mode == "interaction":
        results = results[results["Evaluation"].isin(ho_interaction)]
    else:
        results = results[results["Evaluation"].isin(ho_org)]
    # For MAE and MAPE, compute 95% confidence intervals for the metric.
    # Otherwise, just compute the mean.
    # Compute group statistics: mean, standard deviation, and count.
    agg_data = (
        results.groupby(["Method", "Model"])
        .agg(
            mean_metric=(metric, "mean"),
            std_metric=(metric, "std"),
            count_metric=(metric, "count"),
        )
        .reset_index()
    )
    # Compute the 95% CI half-width:
    # For each group, CI = t(0.975, df=n-1) * (std/sqrt(n)).
    agg_data["ci"] = agg_data.apply(
        lambda row: stats.t.ppf(0.975, row["count_metric"] - 1)
        * row["std_metric"]
        / np.sqrt(row["count_metric"])
        if row["count_metric"] > 1
        else 0,
        axis=1,
    )
    # Sort data by the mean value for plotting
    agg_data = agg_data.sort_values("mean_metric")
    y_value = "mean_metric"
    error_column = "ci"

    # Compute overall best (lowest) value of the metric (using mean_metric when available)
    best_val = agg_data[y_value].min()
    best_model = agg_data["Model"][agg_data[y_value] == best_val].iloc[0]
    best_method = agg_data["Method"][agg_data[y_value] == best_val].iloc[0]

    # Set publication-grade style settings
    sns.set_theme(style="whitegrid", font="serif", font_scale=1)
    plt.rcParams.update(
        {"font.size": 8, "font.family": "sans-serif", "font.weight": "bold"}
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))

    # Draw barplot with seaborn. For MAE/MAPE, disable built-in error bars.

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

    # Add error bars (for MAE/MAPE only) using the computed 95% CI.
    for patch, ci in zip(ax.patches, agg_data[error_column].values):
        x_center = patch.get_x() + patch.get_width() / 2.0
        height = patch.get_height()
        ax.errorbar(x_center, height, yerr=ci, color="black", capsize=5, fmt="none")

    # Add a horizontal dotted line at the overall best value and label it.
    ax.axhline(
        y=best_val,
        linestyle=":",
        color="red",
        linewidth=2,
        label=f"overall best: {best_val:.3f}",
    )

    # Add title and axis labels with improved formatting.
    ax.set_title(
        f"Model Selection Performance ({metric})\n Best Model: {best_model} | Best Method: {best_method}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric, fontsize=14, fontweight="bold")

    # Create a custom legend entry for the overall best line.
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
                + f"_{mode}"
                + save_path[save_path.index(".pdf") :]
            )
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path + f"_{mode}" + ".pdf", format="pdf", bbox_inches="tight"
            )
    if show:
        plt.show()


def summary_model_selection(
    results_file_path, metric="RMSE", method="avg", save_path=None, show=False
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
      save_path : str, optional
          If provided, the plot is saved as a PDF.
      show : bool, optional
          If True, the plot is displayed.

    Note:
      This function assumes that the following lists are defined in the global scope:
          ho_bacillus, ho_pathogen, ho_interaction
      Overall is computed on the union of these sets.
    """
    # Validate metric
    assert metric in [
        "RMSE",
        "MAE",
        "std_abs_err",
        "MAPE",
        "std_abs_relative_err",
        "R2",
    ], "metric must be one of: RMSE, MAE, std_abs_err, MAPE, std_abs_relative_err, R2"

    # Read the CSV file and filter by the selected method.
    results = pd.read_csv(results_file_path)
    results = results[results["Method"] == method]

    # Define the union of evaluation sets for overall performance.
    ho_all = ho_bacillus + ho_pathogen + ho_interaction

    # Helper function to compute aggregated statistics for a given subset.
    def agg_stats(df, eval_type):
        grp = (
            df.groupby("Model")
            .agg(
                mean_metric=(metric, "mean"),
                std_metric=(metric, "std"),
                count_metric=(metric, "count"),
            )
            .reset_index()
        )
        grp["ci"] = grp.apply(
            lambda row: stats.t.ppf(0.975, row["count_metric"] - 1)
            * row["std_metric"]
            / np.sqrt(row["count_metric"])
            if row["count_metric"] > 1
            else 0,
            axis=1,
        )
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
        f"Method: {method.upper()} | Best Model (Overall): {best_model}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric, fontsize=14, fontweight="bold")

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
                + f"_{method}"
                + save_path[save_path.index(".pdf") :]
            )
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path + f"_{method}" + ".pdf", format="pdf", bbox_inches="tight"
            )

    if show:
        plt.show()


def summary_preprocess_selection(
    results_file_path,
    metric="RMSE",
    method="avg",
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
      save_path : str, optionnel
          Si fourni, le graphique sera sauvegardé au format PDF.
      save_agg_data : bool, optionnel
          Si True, les données agrégées seront également sauvegardées au format CSV.
      ci_normalized : bool, optionnel
          Si True, la CI est calculée en normalisant par la valeur moyenne.
      show : bool, optionnel
          Si True, le graphique est affiché.
    """
    # Validation de la métrique
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

    # Lecture du CSV et filtrage selon la méthode
    results = pd.read_csv(results_file_path)
    results = results[results["Method"] == method]

    # Ensemble des évaluations pour le groupe Overall
    ho_all = ho_bacillus + ho_pathogen + ho_interaction

    # Fonction d'agrégation pour un sous-ensemble d'évaluations.
    def agg_stats(df, eval_type, ci_normalized):
        grp = (
            df.groupby("Model")
            .agg(
                mean_metric=(metric, "mean"),
                std_metric=(metric, "std"),
                count_metric=(metric, "count"),
            )
            .reset_index()
        )
        if ci_normalized:
            grp["ci"] = grp.apply(
                lambda row: row["mean_metric"]
                / (
                    stats.t.ppf(0.975, row["count_metric"] - 1)
                    * row["std_metric"]
                    / np.sqrt(row["count_metric"])
                )
                if row["count_metric"] > 1
                else 0,
                axis=1,
            )
        else:
            grp["ci"] = grp.apply(
                lambda row: stats.t.ppf(0.975, row["count_metric"] - 1)
                * row["std_metric"]
                / np.sqrt(row["count_metric"])
                if row["count_metric"] > 1
                else 0,
                axis=1,
            )
        grp["EvalType"] = eval_type
        return grp

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
        "gpuKNNImputer",
        "gpuRandomForestImputer",
    ]:
        for scaler in [
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "MaxAbsScaler",
        ]:
            order_models.append(f"{imputer}_{scaler}")
    available_models = [m for m in order_models if m in agg_data["Model"].unique()]

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
        best_idx = overall_group["ratio"].idxmin()
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
    fig, ax = plt.subplots(figsize=(15, 7))

    # Sauvegarde des données agrégées si demandé.
    if save_agg_data:
        if save_path is not None:
            if save_path.endswith(".pdf"):
                save_agg_path = (
                    save_path[: save_path.index(".pdf")] + f"_{method}_agg_data.csv"
                )
            else:
                save_agg_path = f"_{method}_agg_data.csv"
        else:
            save_agg_path = f"_{method}_agg_data.csv"
        agg_data.to_csv(save_agg_path, index=False)

    # Tracé des points avec barres d'erreur
    # On définit des offsets pour décaler les points par type d'évaluation (pour éviter qu'ils ne se superposent)
    dodge_offsets = {
        "Overall": -0.15,
        "Bacillus": -0.05,
        "Pathogen": 0.05,
        "Interaction": 0.15,
    }
    # Mapping des modèles aux positions sur l'axe des x
    model_to_x = {model: i for i, model in enumerate(available_models)}

    # Pour chaque type d'évaluation, tracer le point et son intervalle de confiance
    for eval_type in eval_order:
        sub = agg_data[agg_data["EvalType"] == eval_type]
        # Calcul des positions x décalées
        x_positions = [model_to_x[m] + dodge_offsets[eval_type] for m in sub["Model"]]
        ax.errorbar(
            x_positions,
            sub["mean_metric"],
            yerr=sub["ci"],
            fmt="o",
            capsize=5,
            markersize=8,
            label=eval_type,
            linestyle="None",
            markeredgecolor="black",
        )

    # Ajout d'une ligne horizontale pointillée indiquant la meilleure stratégie (selon le ratio)
    if best_metric is not None:
        ax.axhline(
            y=best_metric,
            linestyle=":",
            color="red",
            linewidth=2,
            label=f"Best Overall ({best_model}): {best_metric:.3f} (ratio: {best_ratio:.3f})",
        )

    ax.set_title(
        f"Résumé de la sélection de preprocessing ({metric} ± 95% CI)\n"
        f"Méthode: {method.upper()} | Meilleure stratégie: {best_model}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Preprocessing", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric, fontsize=14, fontweight="bold")

    # Positionnement et étiquetage de l'axe des x
    ax.set_xticks(range(len(available_models)))
    ax.set_xticklabels(available_models, rotation=15)

    ax.legend(title="Type d'évaluation", title_fontsize=12, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith(".pdf"):
            save_path_modified = (
                save_path[: save_path.index(".pdf")]
                + f"_{method}"
                + save_path[save_path.index(".pdf") :]
            )
            plt.savefig(save_path_modified, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_path + f"_{method}.pdf", format="pdf", bbox_inches="tight")

    if show:
        plt.show()


def plot_model_and_dataset_selection(metric):
    assert metric in ["mae", "mape", "rmse", "r2"], (
        "metric must be in the following ['mae', 'mape', 'rmse', 'r2']"
    )

    CV_FILES = [
        f for f in os.listdir("./Results/model_selection") if f.startswith("cv_")
    ]
    MODELS = [f[3 : f[3:].index("_") + 3] for f in CV_FILES]
    print(CV_FILES)
    print(MODELS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["MAE", "RMSE"],
        help="Which metric to plot. defaults is MAE and RMSE",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["avg", "random", "combinatoric"],
        help="Which methods to plot. defaults is MAE and RMSE",
    )
    parser.add_argument(
        "--plot_model_selection",
        default=1,
        help="Plot model selection (1=default) or not (0) ",
    )
    parser.add_argument(
        "--plot_preprocess_selection",
        default=1,
        help="Plot preprocess selection (1=default) or not (0) ",
    )

    args = parser.parse_args()

    for metric in args.metrics:
        if args.plot_model_selection:
            plot_model_selection(
                "./Results/model_selection/ho_all_results.csv",
                metric,
                mode="all",
                save_path=f"./Plots/model_selection_{metric}.pdf",
            )
        for method in args.methods:
            if args.plot_model_selection:
                summary_model_selection(
                    "./Results/model_selection/ho_all_results.csv",
                    metric=metric,
                    method=method,
                    save_path=f"./Plots/summary_model_selection_{metric}.pdf",
                    show=False,
                )
            if args.plot_preprocess_selection:
                summary_preprocess_selection(
                    "./Results/preprocess_selection/ho_all_results.csv",
                    metric,
                    method=method,
                    save_path=f"./Plots/preprocess_selection_{metric}.pdf",
                )
