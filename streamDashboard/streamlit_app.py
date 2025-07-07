# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
import textwrap

st.set_page_config(layout="wide", page_title="Sector Dashboard")
#Cache to run once per session
@st.cache_data
def load_and_filter():
    itac = pd.read_csv('itac_compact.csv').dropna(subset=['SIC'])
    itac['SIC'] = itac['SIC'].astype(int)
    sic = pd.read_csv('sic-codes.csv')
    sic['SIC'] = sic['SIC'].astype(int)

    df = (
        itac.merge(sic[['SIC','Description','Major Group']], on='SIC', how='left')
             .query("FY >= 2018 and STATE == 'CA'"
                    " and 100_000 < ANNUAL_ELECTRICITY_COST < 3_000_000"
                    " and TOTAL_RECC_SAVINGS >= 100_000"
                    " and ROI_YRS <= 5")
             .reset_index(drop=True)
    )
    return df

df = load_and_filter()

# ---- Page functions ----

def sic_lookup_page(df):
    st.title("🔍 SIC Lookup")
    lookup = (
        df[['SIC','Description']]
        .drop_duplicates('SIC')
        .rename(columns={'Description':'Industry'})
        .sort_values('SIC')
        .reset_index(drop=True)
    )
    search = st.text_input("Search SIC or Industry:")
    if search:
        mask = (
            lookup['Industry'].str.contains(search, case=False, na=False) |
            lookup['SIC'].astype(str).str.contains(search, na=False)
        )
        lookup = lookup[mask]
    st.dataframe(lookup, use_container_width=True)
#-----------------------------------------------SIC-----------------------------------------



def single_sic_comparison_page(df):
    st.title("📊 Single SIC Comparison with Global Metrics")

    # 1) SIC selector
    sic_list = sorted(df['SIC'].unique())
    sic_code = st.sidebar.selectbox("Select SIC code", sic_list)
    sect = df[df['SIC'] == sic_code].copy()
    if sect.empty:
        st.error("No data for that SIC after filtering.")
        return

    # Lookup industry name
    desc = df.drop_duplicates('SIC').set_index('SIC')['Description']
    industry = desc.get(sic_code, "Unknown Industry")
    st.header(f"SIC {sic_code}: {industry}")
    st.markdown(f"**Filtered records:** {len(sect)}")

    # 2) Scale & rename
    raw = {
        'ANNUAL_SALES':            ('Annual Sales (M)',        1e6),
        'EMPLOYEES':               ('Employees',               1),
        'PLANT_AREA':              ('Plant Area (M sqft)',     1e6),
        'ANNUAL_ELECTRICITY_COST': ('Electricity Cost (M)',    1e6),
        'TOTAL_RECC_IMP_COST':     ('Implementation Cost (M)', 1e6),
        'TOTAL_RECC_SAVINGS':      ('Savings (M)',             1e6),
        'ROI_YRS':                 ('ROI (yrs)',               1),
    }
    cols = [c for c in raw if c in df.columns]
    sect_scaled = pd.DataFrame({
        label: sect[col]/scale
        for col,(label,scale) in raw.items() if col in sect
    })

    # 3) Summary stats
    stats = sect_scaled.describe().T[['count','mean','std','25%','50%','75%']]
    st.subheader(f"{industry} Summary (scaled)")
    st.dataframe(stats.style.format("{:.2f}"))

    # 4) Comparison table
    global_scaled = pd.DataFrame({
        label: df[col]/scale
        for col,(label,scale) in raw.items() if col in df
    })
    gmean = global_scaled.mean()
    gmed  = global_scaled.median()

    comp = pd.DataFrame({
        f'{industry} Mean':  stats['mean'],
        'Global Mean':       gmean,
        'Mean Diff':         stats['mean'] - gmean,
        f'{industry} Median':stats['50%'],
        'Global Median':     gmed,
        'Median Diff':       stats['50%'] - gmed,
    })

    st.subheader(f"{industry} vs Global Comparison")
    st.dataframe(comp.style.format("{:.2f}"))

# 5) Distributions (2×4 hist + optional KDE + mean & median lines)
    st.subheader(f"{industry} Metrics Distributions")
    sns.set_theme(style="whitegrid")

    labels = list(sect_scaled.columns)
    n      = len(labels)
    ncols  = min(4, n)
    nrows  = (n + ncols - 1)//ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, lab in enumerate(labels):
        data = sect_scaled[lab].dropna()
        ax   = axes[i]

        # compute bins via Freedman–Diaconis, fallback for single-value
        if data.nunique() <= 1:
            v    = data.iloc[0]
            bins = [v - 1, v + 1]
        else:
            bins = np.histogram_bin_edges(data, bins="fd")

        # histogram
        counts, edges, _ = ax.hist(
            data,
            bins=bins,
            color="#FFA500",
            edgecolor="black",
            alpha=0.6
        )

        # KDE only if >1 point
        if len(data) > 1:
            try:
                kde = gaussian_kde(data)
                xs  = np.linspace(data.min(), data.max(), 200)
                bw  = edges[1] - edges[0]
                ax.plot(xs, kde(xs)*len(data)*bw, color="#FF8C00", lw=2)
            except Exception:
                pass

        # mean line + annotation
        m = data.mean()
        ax.axvline(m, linestyle="--", color="gray", lw=1)
        ax.text(
            m, ax.get_ylim()[1]*0.9, f"μ={m:.2f}",
            ha="center", va="bottom", color="gray", fontsize=9
        )

        # median line + annotation
        me = data.median()
        ax.axvline(me, linestyle="-.",
               color="black", lw=1.5)
        ax.text(
            me, ax.get_ylim()[1]*0.75, f"Med={me:.2f}",
            ha="center", va="bottom", color="black", fontsize=9
        )

        # formatting
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    # remove any empty subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # 6) Global vs Sector bar charts per metric (mean & median only)
    st.subheader(f"Global vs {industry} Mean & Median")

    for metric in comp.index:
        # pick just the four values in the right order
        keys = ["Global Mean", f"{industry} Mean", "Global Median", f"{industry} Median"]
        vals = comp.loc[metric, keys].astype(float)

        # make the figure
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#555555", "#FFA500", "#AAAAAA", "#D62728"]
        x = np.arange(len(keys))

        bars = ax.bar(x, vals.values, color=colors, edgecolor="black", lw=1)

        # formatting
        ax.set_title(f"{metric}: Global vs. {industry}", fontsize=14, pad=14)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, vals.max() * 1.15)

        # annotate
        top = vals.max() * 0.02
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + top,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)


#---------------------------------------Multi SIC comparisons-------------------------------------
def multi_sic_clustering_page(df):
    st.title("🔍 Compare Multiple Sectors")

    # 1) SIC selector
    all_sics = df['SIC'].dropna().astype(int).unique()
    sic_list = st.multiselect(
        "Select one or more SIC codes:",
        options=sorted(all_sics),
        format_func=lambda s: f"{s} – {df.loc[df['SIC']==s, 'Description'].iloc[0]}",
    )
    if not sic_list:
        st.info("Please select at least one SIC code to compare.")
        return

    # 2) Metric definitions
    raw = {
        'ANNUAL_SALES':            ('Annual Sales (M USD)',       1e6),
        'EMPLOYEES':               ('Employees',                  1),
        'PLANT_AREA':              ('Plant Area (M sqft)',        1e6),
        'ANNUAL_ELECTRICITY_COST': ('Electricity Cost (M USD)',   1e6),
        'TOTAL_RECC_IMP_COST':     ('Implementation Cost (M USD)',1e6),
        'TOTAL_RECC_SAVINGS':      ('Savings (M USD)',            1e6),
        'ROI_YRS':                 ('ROI (yrs)',                  1),
    }

    # 3) Compute global mean & median
    global_scaled = {
        label: df[col] / scale
        for col,(label,scale) in raw.items() if col in df
    }
    gmean = pd.Series({lbl: vals.mean() for lbl, vals in global_scaled.items()})
    gmed  = pd.Series({lbl: vals.median() for lbl, vals in global_scaled.items()})

    # 4) Compute per-sector mean & median
    desc_lookup = df.drop_duplicates('SIC').set_index('SIC')['Description']
    data_means = {"Global": gmean}
    data_meds  = {"Global": gmed}

    for sic in sic_list:
        sect = df[df['SIC']==sic]
        scaled = {
            label: sect[col] / scale
            for col,(label,scale) in raw.items() if col in sect
        }
        name = f"{sic} – {desc_lookup.get(sic,'')}"
        data_means[name] = pd.Series({lbl: vals.mean()   for lbl, vals in scaled.items()})
        data_meds[name]  = pd.Series({lbl: vals.median() for lbl, vals in scaled.items()})

    means_df   = pd.DataFrame(data_means)
    medians_df = pd.DataFrame(data_meds)

    # 5) Plot Means
    st.subheader("📊 Mean Comparison")
    fig, axes = plt.subplots(2, 4, figsize=(40,20),constrained_layout=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, means_df.index):
        vals = means_df.loc[metric]
        bars = ax.bar(
            vals.index, vals.values,
            color=sns.color_palette("tab10", len(vals)),
            edgecolor="black"
        )
        ax.set_title(metric, fontsize=17)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=45, ha="right", fontsize=14)
        ylim = vals.max() * 1.2
        ax.set_ylim(0, ylim)
        for bar, v in zip(bars, vals.values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                v + ylim*0.03,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=15
            )
    # remove unused subplots
    for ax in axes[len(means_df.index):]:
        fig.delaxes(ax)
    st.pyplot(fig)

    # 6) Plot Medians
    st.subheader("📊 Median Comparison")
    fig, axes = plt.subplots(2, 4, figsize=(40,20))
    axes = axes.flatten()
    for ax, metric in zip(axes, medians_df.index):
        vals = medians_df.loc[metric]
        bars = ax.bar(
            vals.index, vals.values,
            color=sns.color_palette("Set2", len(vals)),
            edgecolor="black"
        )
        ax.set_title(metric, fontsize=17)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=45, ha="right", fontsize=14)
        ylim = vals.max() * 1.2
        ax.set_ylim(0, ylim)
        for bar, v in zip(bars, vals.values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                v + ylim*0.03,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=15
            )
    for ax in axes[len(medians_df.index):]:
        fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig)

    # 7) Dendrogram
    st.subheader("🌳 Sector Clustering (Euclidean Ward)")
    prof = (
        df[df['SIC'].isin(sic_list)]
          .assign(**{col: df[col]/scale for col,(lbl,scale) in raw.items()})
          .groupby('SIC')[[c for c in raw]]
          .mean()
    )
    X = StandardScaler().fit_transform(prof.values)
    dvec = pdist(X, metric='euclidean')
    link = hierarchy.linkage(dvec, method='ward')

    labels = [
        textwrap.shorten(f"{s}\n{desc_lookup.get(s,'')}", width=30, placeholder="…")
        for s in prof.index
    ]
    plt.figure(figsize=(14,6))
    hierarchy.dendrogram(
        link, labels=labels, leaf_rotation=45, leaf_font_size=10,
        orientation="top", color_threshold=0
    )
    plt.ylabel("Distance")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # 8) Correlation Clustermap
    st.subheader("🔗 Sector Similarity: Pearson Correlation")
    corr = prof.T.corr()
    cg = sns.clustermap(
        corr, cmap="vlag", center=0, linewidths=1,
        figsize=(12,12), annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Pearson r"}
    )
    cg.fig.suptitle("Correlation between Sector Metric Vectors", y=1.02)
    st.pyplot(cg.fig)

# ---- Page registry ----

PAGES = {
    "SIC Lookup": sic_lookup_page,
    "Single SIC Comparison": single_sic_comparison_page,
    "Multi-SIC Clustering": multi_sic_clustering_page,
}

st.sidebar.title("📄 Pages")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
# execute the selected page
PAGES[selection](df)





