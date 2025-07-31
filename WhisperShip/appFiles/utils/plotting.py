import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sns.set_theme(style="whitegrid")

def plot_hist_kde(data, title):
    """
    Histogram + Freedman–Diaconis KDE + mean/median lines.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    # bins
    if data.nunique() <= 1:
        v = float(data.iloc[0])
        bins = [v - 1, v + 1]
    else:
        bins = np.histogram_bin_edges(data, bins="fd")

    ax.hist(data, bins=bins, color="#FFA500", edgecolor="black", alpha=0.6)

    if len(data) > 1:
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 200)
        bw = bins[1] - bins[0]
        ax.plot(xs, kde(xs) * len(data) * bw, color="#FF8C00", lw=2)

    m  = data.mean()
    md = data.median()
    ax.axvline(m,  color="gray", linestyle="--", lw=1)
    ax.axvline(md, color="black",linestyle="-.", lw=1.5)

    ax.text(m,  ax.get_ylim()[1]*0.9,  f"μ={m:.2f}", color="gray", ha="center")
    ax.text(md, ax.get_ylim()[1]*0.75, f"Med={md:.2f}", color="black",ha="center")

    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.tight_layout()
    return fig

def hbar_compare(vals, labels, title, xlabel, palette="tab10", figsize=(7,4)):
    """
    Horizontal bar plot with annotations.
    vals: array‐like of values
    labels: list of index labels
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=vals, y=labels, palette=palette, edgecolor="black", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for p in ax.patches:
        ax.text(
            p.get_width() * 1.01,
            p.get_y() + p.get_height()/2,
            f"{p.get_width():.2f}",
            va="center"
        )
    plt.tight_layout()
    return fig

