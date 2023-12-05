import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

output_dir = r"C:\Users\tim\Documents\tex\dissertation\all_in\musicreview_figures"

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

for p in [0.24, 0.59]:
    c = 0.1
    e = np.linspace(0, 1, 150)

    e125 = np.linspace(0.1, 1, 150)
    y125 = 1 - (e125 * p * (1 - c) + c) / (p * (1 - c) + c)
    # e125 = np.concatenate([[0, e125[0]], e125])
    # y125 = np.concatenate([[0, 0], y125])

    e25 = np.linspace(0.25, 1, 150)
    c = 0.25
    y25 = 1 - (e25 * p * (1 - c) + c) / (p * (1 - c) + c)
    # e25 = np.concatenate([[0, e25[0]], e25])
    # y25 = np.concatenate([[0, 0], y25])

    e5 = np.linspace(0.5, 1, 150)
    c = 0.5
    y50 = 1 - (e5 * p * (1 - c) + c) / (p * (1 - c) + c)
    # e5 = np.concatenate([[0, e5[0]], e5])
    # y50 = np.concatenate([[0, 0], y50])

    fig = plt.figure(figsize=(7, 4))
    plt.plot(e5, y50, label="$c$ = 0.50")
    plt.plot(e25, y25, label="$c$ = 0.25")
    plt.plot(e125, y125, label="$c$ = 0.1")

    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1])
    plt.xticks(np.arange(0, 1, 0.1))

    plt.legend()

    plt.ylabel("Time Saved by Error Detector")
    plt.xlabel("Amount of score marked as erroneous by detector $e$")

    fig.savefig(
        os.path.join(output_dir, f"epc_diagram_p{p}.pdf"),
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    plt.clf()

    # all_df = pd.DataFrame(
    #     {
    #         "$e$": e,
    #         "$c$ = 0.125": y125,
    #         "$c$ = 0.25": y25,
    #         "$c$ = 0.50": y50,
    #     }
    # ).melt("$e$")

    # g = sns.lineplot(
    #     data=all_df, x="$e$", y="value", palette="colorblind", hue="variable"
    # )
    # g.legend(title="Value for $c$")
    # g.set(
    #     ylabel="Time Saved by Error Detector",
    #     xlabel="Amount of score marked as erroneous by detector $e$",
    # )
    # g.set_xlim([0, 1.01])
    # g.set_ylim([0.0, 1.01])
    # fig = g.get_figure()
