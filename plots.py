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


def plot_model_selection(
    results_file_path,
    metric="RMSE",
    mode="all",
    avg_mode="unweighted",
    save_path=None,
    show=False,
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
        # Compute weighted mean
        agg_data = (
            results.groupby(["Method", "Model"])
            .apply(
                lambda g: np.sum(g[metric] * g["n_samples"]) / np.sum(g["n_samples"])
            )
            .reset_index(name="mean_metric")
        )
    else:
        # Compute unweighted mean
        agg_data = (
            results.groupby(["Method", "Model"])
            .agg(mean_metric=(metric, "mean"))
            .reset_index()
        )

    # Compute standard deviation and count for confidence intervals
    agg_stats = (
        results.groupby(["Method", "Model"])
        .agg(
            std_metric=(metric, "std"),
            count_metric=(metric, "count"),
        )
        .reset_index()
    )

    # Merge statistics into agg_data
    agg_data = pd.merge(agg_data, agg_stats, on=["Method", "Model"])

    # Compute the 95% CI half-width:
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
                            "std_metric": g[metric].std(),
                            "count_metric": g[metric].count(),
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
                            "std_metric": g[metric].std(),
                            "count_metric": g[metric].count(),
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
                    std_metric=(metric, "std"),
                    count_metric=(metric, "count"),
                )
                .reset_index()
            )
        # Calcul de l'intervalle de confiance à 95%.
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
                save_agg_path = f"_{method}_{avg_mode}_agg_data.csv"
        else:
            save_agg_path = f"_{method}_{avg_mode}_agg_data.csv"
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
            label=f"Best Overall ({best_model}): {best_metric:.3f} \n({metric} / CI ratio: {best_ratio:.3f})",
        )

    ax.set_title(
        f"Summary of preprocessing selection ({metric} ± 95% CI)\n"
        f"Method: {method.upper()} | Averaging: {avg_mode.capitalize()} | Best preprocessing: {best_model}",
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


import matplotlib.cm as cm


def plot_feature_selection(folder_path, save_path):
    """
    Plot a bar histogram showing permutation-based feature selection performance.

    For each CSV in folder_path, the function computes the weighted mean of RMSE and MAE
    for each permuted feature. The plot shows the performance change during the selection process.
    It also generates step-wise plots showing the importance of each feature in each step.

    Parameters:
        folder_path : str
            Path to the folder containing CSV files.
        save_path : str
            Base save path (the figure will be saved as f"{save_path}_feature_selection.pdf").
    """
    csv_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".csv") and "summary" in f
        ]
    )
    if not csv_files:
        raise ValueError("No summary CSV files found in the specified folder.")

    step_names = []
    performance_values = []
    colors = []
    first_color = "blue"
    best_color = "green"
    final_color = "red"
    default_color = "lightgray"
    best_intermediate_idx = None

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        if "Weighted Cross Mean (RMSE and MAE)" not in df.columns:
            raise ValueError(f"Weighted Cross Mean column not found in file {file}.")

        if i == 0:
            step_names.append("All features")
            performance_values.append(df["Weighted Cross Mean (RMSE and MAE)"].iloc[0])
            colors.append(first_color)
        else:
            best_val = df["Weighted Cross Mean (RMSE and MAE)"].min()
            removed = df["Permutation"][
                df["Weighted Cross Mean (RMSE and MAE)"] == best_val
            ].iloc[0]
            step_names.append(f"Step {i} Removed: {removed}")
            performance_values.append(best_val)
            colors.append(default_color)

        # Generate step-wise plot
        plt.figure(figsize=(10, 6))

        # Sort values so that "No Permutation" is first
        df = df.sort_values(
            by="Permutation", key=lambda x: x == "No Permutation", ascending=False
        )

        num_features = len(df["Permutation"])

        if num_features > 15:
            # Horizontal bar plot for large feature sets
            plt.figure(
                figsize=(10, num_features * 0.5)
            )  # Adjust figure size based on feature count
            bars = plt.barh(
                df["Permutation"],
                df["Weighted Cross Mean (RMSE and MAE)"],
                edgecolor="black",
            )
            plt.yticks(fontsize=8)
            plt.xlabel("Permutation Feature Importance", fontsize=12)
            plt.ylabel("Features", fontsize=12)
        else:
            # Standard bar plot for smaller feature sets
            bars = plt.bar(
                df["Permutation"],
                df["Weighted Cross Mean (RMSE and MAE)"],
                edgecolor="black",
            )
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.ylabel("Permutation Feature Importance", fontsize=12)
            plt.xlabel("Features", fontsize=12)

        # Color coding
        inferno = cm.get_cmap("inferno")
        no_perm_color = inferno(0.2)
        best_perm_color = inferno(0.8)

        for bar, perm in zip(bars, df["Permutation"]):
            if perm == "No Permutation":
                bar.set_color(no_perm_color)
            elif (
                perm
                == df["Permutation"][df["Weighted Cross Mean (RMSE and MAE)"].idxmin()]
            ):
                bar.set_color(best_perm_color)
            else:
                bar.set_color("lightgray")

        plt.title(f"Permutation Importance Step {i}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/step_{i + 1}_importance.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

    if len(performance_values) > 2:
        intermediate_vals = performance_values[1:]
        best_intermediate_idx = np.argmin(intermediate_vals) + 1
        colors[best_intermediate_idx] = best_color

    colors[-1] = final_color

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(step_names)), performance_values, color=colors, edgecolor="black"
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(i, height, f"{height:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(range(len(step_names)), step_names, rotation=45, ha="right", fontsize=12)
    plt.ylabel("Weighted Cross Mean (RMSE and MAE)", fontsize=14)
    plt.xlabel("Feature Selection Step", fontsize=14)
    plt.title(
        "Permutation-Based Feature Selection Performance",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    save_filename = f"{save_path}/permutation_feature_selection.pdf"
    plt.savefig(save_filename, format="pdf", bbox_inches="tight")
    plt.close()


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

    for avg_mode in ["weighted", "unweighted"]:
        for metric in args.metrics:
            if args.plot_model_selection:
                plot_model_selection(
                    "./Results/model_selection/ho_all_results.csv",
                    metric,
                    mode="all",
                    avg_mode=avg_mode,
                    save_path=f"./Plots/model_selection_{metric}.pdf",
                )
            for method in args.methods:
                if args.plot_model_selection:
                    summary_model_selection(
                        "./Results/model_selection/ho_all_results.csv",
                        metric=metric,
                        method=method,
                        avg_mode=avg_mode,
                        save_path=f"./Plots/summary_model_selection_{metric}.pdf",
                        show=False,
                    )
                if args.plot_preprocess_selection:
                    summary_preprocess_selection(
                        "./Results/preprocess_selection/ho_all_results.csv",
                        metric,
                        avg_mode=avg_mode,
                        method=method,
                        save_path=f"./Plots/preprocess_selection_{metric}.pdf",
                    )
