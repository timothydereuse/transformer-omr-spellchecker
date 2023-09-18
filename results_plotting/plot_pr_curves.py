import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

output_dir = r"C:\Users\tim\Documents\tex\dissertation\results"

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
    "figure.figsize": (7, 4),
}
plt.rcParams.update(params)

# PRECISION RECALL
df1 = pd.read_csv(r"results_csv\knn_PR_curve.csv").iloc[::50, :].assign(trial="KNN")
df2 = pd.read_csv(r"results_csv\512_PR_curve.csv").iloc[::50, :].assign(trial="4096")
df3 = pd.read_csv(r"results_csv\64_PR_curve.csv").iloc[::50, :].assign(trial="64")

all_df = pd.concat([df1, df2, df3])

plt.rcParams.update({"figure.figsize": (5, 4)})

g = sns.lineplot(data=all_df, x="recall", y="precision", palette="viridis", hue="trial")
g.legend(title="Sequence Length")
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
