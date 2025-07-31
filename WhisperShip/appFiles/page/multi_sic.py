import streamlit as st
import pandas as pd
import textwrap
from utils.data import METRICS, scale_df
from utils.plotting import hbar_compare
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def multi_sic_clustering_page(df: pd.DataFrame):
    st.title("🔍 Compare Multiple SICs")

    # 1) SIC selector
    all_sics = sorted(df['SIC'].dropna().astype(int).unique())
    sic_list = st.multiselect(
        "Select SIC codes",
        options=all_sics,
        format_func=lambda s: f"{s} – {df.loc[df['SIC']==s,'Description'].iat[0]}"
    )
    if not sic_list:
        return st.info("Pick at least one SIC.")

    # 2) desc_lookup for labeling
    desc_lookup = df.drop_duplicates('SIC').set_index('SIC')['Description']

    # 3) Build scaled DataFrames
    #    - global
    global_s = scale_df(df)                          # uses your METRICS
    data_means = {"Global": global_s.mean()}
    data_meds  = {"Global": global_s.median()}

    #    - per SIC
    for sic in sic_list:
        sect = df[df['SIC']==sic]
        sect_s = scale_df(sect)
        name   = f"{sic} – {desc_lookup[sic]}"
        data_means[name] = sect_s.mean()
        data_meds[name]  = sect_s.median()

    means_df   = pd.DataFrame(data_means)
    medians_df = pd.DataFrame(data_meds)

    # 4) Mean grid
    st.subheader("Mean Comparison")
    fig, axes = plt.subplots(2, 4, figsize=(20,10), constrained_layout=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, means_df.index):
        vals = means_df.loc[metric].values
        labs = means_df.columns
        bars = ax.bar(labs, vals,
                      color=sns.color_palette("tab10", len(vals)),
                      edgecolor="black")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.01, f"{v:.2f}", ha='center')
    # drop extras
    for extra in axes[len(means_df.index):]:
        fig.delaxes(extra)
    st.pyplot(fig)

    # 5) Median grid
    st.subheader("Median Comparison")
    fig2, axes2 = plt.subplots(2, 4, figsize=(20,10), constrained_layout=True)
    axes2 = axes2.flatten()
    for ax, metric in zip(axes2, medians_df.index):
        vals = medians_df.loc[metric].values
        labs = medians_df.columns
        bars = ax.bar(labs, vals,
                      color=sns.color_palette("Set2", len(vals)),
                      edgecolor="black")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.01, f"{v:.2f}", ha='center')
    for extra in axes2[len(medians_df.index):]:
        fig2.delaxes(extra)
    st.pyplot(fig2)

    # 6) Prepare for clustering

    df_scaled = df.copy()
    df_scaled['ANNUAL_SALES']           /= 1e6  # now in M USD
    df_scaled['PLANT_AREA']             /= 1e6  # now in M sqft
    df_scaled['ANNUAL_ELECTRICITY_COST']/= 1e6  # now in M USD
    df_scaled['TOTAL_RECC_SAVINGS']     /= 1e6  # now in M USD
    df_scaled['TOTAL_RECC_IMP_COST']    /= 1e6  # now in M USD

    prof = (
        df_scaled[df_scaled['SIC'].isin(sic_list)]
          .groupby('SIC')[list(METRICS.keys())]
          .mean()
    )

    # guard single‐SIC case
    if prof.shape[0] < 2:
        st.warning("Select at least two SIC codes to view clustering & similarity plots.")
        return

    # 7) Dendrogram
    X     = StandardScaler().fit_transform(prof.values)
    dvec  = pdist(X, metric='euclidean')
    link  = hierarchy.linkage(dvec, method='ward')
    labels = [
        f"{s}\n{textwrap.shorten(desc_lookup[s], width=30, placeholder='…')}"
        for s in prof.index
    ]
    fig3 = plt.figure(figsize=(14,6))
    hierarchy.dendrogram(link,
                         labels=labels,
                         leaf_rotation=45,
                         leaf_font_size=10,
                         orientation="top")
    plt.ylabel("Ward Distance")
    plt.tight_layout()
    st.pyplot(fig3)

    # 8) Pearson clustermap
    corr = prof.T.corr()
    cg = sns.clustermap(
        corr,
        cmap="vlag",
        center=0,
        linewidths=1,
        figsize=(12,12),
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={'size':8},
    )
    cg.fig.suptitle("Sector Similarity: Pearson Correlation", y=1.02)
    st.pyplot(cg.fig)
