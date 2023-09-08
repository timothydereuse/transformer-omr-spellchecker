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
    "figure.figsize": (9, 5),
}
plt.rcParams.update(params)

df = pd.read_csv("results_csv\seqlen_valnormrecall_training.csv")
dfl = pd.melt(df, "Step")

g = sns.lineplot(data=dfl, x="Step", y="value", palette="viridis", hue="variable")
g.legend(title="Sequence Length")
g.set(xlabel="Epoch", ylabel="Normalized Recall")
g.set_xlim([0, 76])
g.set_ylim([0.75, 1])

fig = g.get_figure()
fig.savefig(
    os.path.join(output_dir, "valnormrecall_seqlen.pdf"),
    bbox_inches="tight",
    pad_inches=0,
    format="pdf",
)
fig.clf()

# VAL LOSS
df = pd.read_csv("results_csv\seqlen_valloss_training.csv")
dfl = pd.melt(df, "Step")

g = sns.lineplot(data=dfl, x="Step", y="value", palette="viridis", hue="variable")
g.legend(title="Sequence Length")
g.set(xlabel="Epoch", ylabel="Loss")
g.set_xlim([0, 76])
g.set_ylim([0.0, 1.06634e-05])
fig = g.get_figure()
fig.savefig(
    os.path.join(output_dir, "valloss_seqlen.pdf"),
    bbox_inches="tight",
    pad_inches=0,
    format="pdf",
)
