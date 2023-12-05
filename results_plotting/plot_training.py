import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

output_dir = r"C:\Users\tim\Documents\tex\dissertation\all_in\results_graphs"
sns.set_context(
    "paper",
)

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
    "figure.figsize": (7, 2.5),
}
plt.rcParams.update(params)
plt.clf()

for fname in ["dataaug", "datasets", "seqlen", "architectures"]:
    df = pd.read_csv(f"results_csv\{fname}_valnormrecall_training.csv")
    df["Step"] = df["Step"] + 1

    if fname != "datasets":
        df_pretrain = df.iloc[:31]
        df_ft = df.iloc[31:]
        df_ft["Step"] = df_ft["Step"] - 31
        df_ft_l = pd.melt(df_ft, "Step")
        df_pretrain_l = pd.melt(df_pretrain, "Step")
        params_sets = [(df_pretrain_l, "pre"), (df_ft_l, "ft")]
    else:
        df_ft_l = pd.melt(df, "Step")
        params_sets = [(df_ft_l, "all")]

    for gparams in params_sets:
        dat, name = gparams
        # NORM RECALL ft
        g = sns.lineplot(
            data=dat,
            x="Step",
            y="value",
            palette="colorblind",
            hue="variable",
        )

        g.legend(title="Trial")
        plt.setp(g.get_legend().get_texts(), fontsize="8")  # for legend text
        plt.setp(g.get_legend().get_title(), fontsize="8")
        # sns.move_legend(g, "lower right")
        g.set(xlabel="Epoch", ylabel="Normalized Recall")
        h, l = g.get_legend_handles_labels()
        g.legend_.remove()
        g.legend(h, l, ncol=2, fontsize="8")

        min_val = df.min().iloc[1:].min() * 0.95
        xbound = dat[pd.notna(dat["value"])]["Step"].max()
        g.set_xlim([1, xbound])
        g.set_ylim([min_val, 1])
        fig = g.get_figure()
        fig.savefig(
            os.path.join(output_dir, f"valnormrecall_{name}_{fname}.pdf"),
            bbox_inches="tight",
            pad_inches=0.05,
            format="pdf",
        )
        fig.clf()

    # VAL LOSS
    df = pd.read_csv(f"results_csv\{fname}_valloss_training.csv")
    df["Step"] = df["Step"] + 1
    if fname != "datasets":
        df_pretrain = df.iloc[:31]
        df_ft = df.iloc[31:]
        df_ft["Step"] = df_ft["Step"] - 31
        df_ft_l = pd.melt(df_ft, "Step")
        df_pretrain_l = pd.melt(df_pretrain, "Step")
        params_sets = [(df_pretrain_l, "pre"), (df_ft_l, "ft")]
    else:
        df_ft_l = pd.melt(df, "Step")
        params_sets = [(df_ft_l, "all")]

    for gparams in params_sets:
        dat, name = gparams
        g = sns.lineplot(
            data=dat,
            x="Step",
            y="value",
            palette="colorblind",
            hue="variable",
        )
        g.legend(title="Trial")
        plt.setp(g.get_legend().get_texts(), fontsize="8")  # for legend text
        plt.setp(g.get_legend().get_title(), fontsize="8")

        h, l = g.get_legend_handles_labels()
        g.legend_.remove()
        g.legend(h, l, ncol=2, fontsize="8")

        # sns.move_legend(g, "lower right")
        g.set(xlabel="Epoch", ylabel="Validation Loss")
        max_val = dat["value"].max() * 1.03
        xbound = dat[pd.notna(dat["value"])]["Step"].max()
        g.set_xlim([1, xbound])
        g.set_ylim([0.0, max_val])
        fig = g.get_figure()
        fig.savefig(
            os.path.join(output_dir, f"valloss_{name}_{fname}.pdf"),
            bbox_inches="tight",
            pad_inches=0.05,
            format="pdf",
        )
        fig.clf()

    # TRAIN LOSS
    df = pd.read_csv(f"results_csv\{fname}_trainloss_training.csv")
    df["Step"] = df["Step"] + 1

    if fname != "datasets":
        df_pretrain = df.iloc[:31]
        df_ft = df.iloc[31:]
        df_ft["Step"] = df_ft["Step"] - 31
        df_ft_l = pd.melt(df_ft, "Step")
        df_pretrain_l = pd.melt(df_pretrain, "Step")
        params_sets = [(df_pretrain_l, "pre"), (df_ft_l, "ft")]
    else:
        df_ft_l = pd.melt(df, "Step")
        params_sets = [(df_ft_l, "all")]

    for gparams in params_sets:
        dat, name = gparams

        g = sns.lineplot(
            data=dat,
            x="Step",
            y="value",
            palette="colorblind",
            hue="variable",
        )
        g.legend(title="Trial")
        plt.setp(g.get_legend().get_texts(), fontsize="8")  # for legend text
        plt.setp(g.get_legend().get_title(), fontsize="8")
        h, l = g.get_legend_handles_labels()
        g.legend_.remove()
        g.legend(h, l, ncol=2, fontsize="8")
        g.set(xlabel="Epoch", ylabel="Training Loss")
        max_val = df.max().iloc[1:].max() * 1.03
        xbound = dat[pd.notna(dat["value"])]["Step"].max()
        g.set_xlim([1, xbound])
        g.set_ylim([0.0, max_val])
        fig = g.get_figure()
        fig.savefig(
            os.path.join(output_dir, f"trainloss_{name}_{fname}.pdf"),
            bbox_inches="tight",
            pad_inches=0.05,
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
