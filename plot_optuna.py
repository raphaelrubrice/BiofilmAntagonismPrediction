import optuna
import matplotlib.pyplot as plt
import pickle as pkl

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_param_importances

if __name__ == "__main__":
    with open("./Results.optuna_campaign/optuna_study.pkl", "rb") as f:
        study = pkl.load(f)

    plot_contour(study)
    plt.savefig("./Plots/optuna_contour.pdf", format="pdf", bbox_inches="tight")

    plot_param_importances(study)
    plt.savefig(
        "./Plots/optuna_param_importances.pdf", format="pdf", bbox_inches="tight"
    )
