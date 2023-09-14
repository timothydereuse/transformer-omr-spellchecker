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
plt.clf()

for fname in [
    "dataaug",
    "seqlen",
]:
    df = pd.read_csv(f"results_csv\{fname}_valnormrecall_training.csv")
    dfl = pd.melt(df, "Step")

    g = sns.lineplot(
        data=dfl,
        x="Step",
        y="value",
        palette="colorblind",
        hue="variable",
    )

    g.legend(title="Trial")
    # sns.move_legend(g, "lower right")
    g.set(xlabel="Epoch", ylabel="Normalized Recall")
    min_val = df.min().iloc[1:].min() * 0.95
    g.set_xlim([0, 75])
    g.set_ylim([min_val, 1])
    fig = g.get_figure()
    fig.savefig(
        os.path.join(output_dir, f"valnormrecall_{fname}.pdf"),
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    fig.clf()

    # VAL LOSS
    df = pd.read_csv(f"results_csv\{fname}_valloss_training.csv")
    dfl = pd.melt(df, "Step")

    g = sns.lineplot(
        data=dfl,
        x="Step",
        y="value",
        palette="colorblind",
        hue="variable",
    )
    g.legend(title="Trial")
    # sns.move_legend(g, "lower right")
    g.set(xlabel="Epoch", ylabel="Loss")
    max_val = df.max().iloc[1:].max()
    g.set_xlim([0, 75])
    g.set_ylim([0.0, max_val])
    fig = g.get_figure()
    fig.savefig(
        os.path.join(output_dir, f"valloss_{fname}.pdf"),
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    fig.clf()

df = pd.read_csv("results_csv\seqlen_lr_training.csv")
df = df[["Step", "64", "512", "1024"]]
dfl = pd.melt(df, "Step")
g = sns.lineplot(
    data=dfl,
    x="Step",
    y="value",
    palette="colorblind",
    hue="variable",
)
g.legend(title="Trial")
g.set(xlabel="Epoch", ylabel="Learning Rate")
max_val = df.max().iloc[1:].max() * 1.02
g.set_xlim([0, 71])
g.set_ylim([0.0, max_val])
fig = g.get_figure()
fig.savefig(
    os.path.join(output_dir, "lr_seqlen.pdf"),
    bbox_inches="tight",
    pad_inches=0,
    format="pdf",
)


fig.clf()
