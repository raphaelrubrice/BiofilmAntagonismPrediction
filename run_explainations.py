from analysis_plots import plot_local_alibi

# --------------------------------------------------
# New interactions: local counterfactuals using Alibi
# --------------------------------------------------

# Local counterfactual explanations using Alibi for a given instance.
# Run for both worst and best cases on model "1234_x_S.en".
plot_local_alibi(
    ho_name="1234_x_S.en",
    mode="worst",
    save_path="./Plots/local/local_alibi_worst.pdf",
    show=False,
)

plot_local_alibi(
    ho_name="1234_x_S.en",
    mode="best",
    save_path="./Plots/local/local_alibi_best.pdf",
    show=False,
)
