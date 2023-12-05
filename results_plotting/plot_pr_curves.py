import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

output_dir = r"C:\Users\tim\Documents\tex\dissertation\all_in\results_graphs"

sns.set_theme()
sns.set_style(style="whitegrid")

params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
    "figure.figsize": (9, 8),
}
plt.rcParams.update(params)

# PRECISION RECALL
df5 = pd.read_csv(r"results_csv\knn_PR_curve.csv").iloc[::50, :].assign(trial="KNN")
df1 = (
    pd.read_csv(r"results_csv\4096_PR_curve.csv")
    .iloc[::50, :]
    .assign(trial="Length: 4096")
)
df0 = pd.read_csv(r"results_csv\512_PR_curve.csv").iloc[::50, :].assign(trial="Base")
df05 = pd.read_csv(r"results_csv\LSTM_PR_curve.csv").iloc[::50, :].assign(trial="LSTM")
df3 = (
    pd.read_csv(r"results_csv\Combined_PR_curve.csv")
    .iloc[::50, :]
    .assign(trial="No Fine-Tuning")
)
df2 = (
    pd.read_csv(r"results_csv\OMRonly_PR_curve.csv")
    .iloc[::50, :]
    .assign(trial="OMR Only")
)
df4 = (
    pd.read_csv(r"results_csv\Synonly_PR_curve.csv")
    .iloc[::50, :]
    .assign(trial="Synth. Only")
)
df6 = pd.read_csv(r"results_csv\UT_PR_curve.csv").iloc[::50, :].assign(trial="UT")


all_df = pd.concat([df0, df05, df1, df2, df3, df4, df5, df6])

g = sns.lineplot(
    data=all_df, x="recall", y="precision", palette="colorblind", hue="trial"
)
g.legend(title="Trial Name")
g.set(ylabel="Precision", xlabel="Recall")
g.set_xlim([0, 1.01])
g.set_ylim([0.0, 1.01])
fig = g.get_figure()
fig.savefig(
    os.path.join(output_dir, "pr_curve_seqlen.pdf"),
    bbox_inches="tight",
    pad_inches=0,
    format="pdf",
)
