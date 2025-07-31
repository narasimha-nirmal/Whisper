import streamlit as st
import pandas as pd
from utils.data import METRICS, scale_df
from utils.plotting import plot_hist_kde, hbar_compare

def single_sic_comparison_page(df: pd.DataFrame):
    st.title("📊 Single SIC Comparison")

    # 1) select SIC
    sic = st.sidebar.selectbox("Select SIC code", sorted(df['SIC'].unique()))
    sect = df[df['SIC'] == sic]
    if sect.empty:
        return st.error("No data for this SIC.")

    # 2) header
    industry = df.loc[df['SIC']==sic,'Description'].iat[0]
    st.header(f"SIC {sic} – {industry} ({len(sect)} records)")

    # 3) scale
    sect_s   = scale_df(sect)
    global_s = scale_df(df)

    # 4) summary stats
    stats = sect_s.describe().T[['count','mean','std','25%','50%','75%']]
    st.subheader("Sector Summary")
    st.dataframe(stats.style.format("{:.2f}"))

    # 5) compare mean/median
    comp = pd.DataFrame({
        f"{industry} Mean":   sect_s.mean(),
        "Global Mean":        global_s.mean(),
        f"{industry} Median": sect_s.median(),
        "Global Median":      global_s.median(),
    })
    st.subheader("Mean & Median Comparison")
    st.dataframe(comp.style.format("{:.2f}"))

    # 6) distributions
    st.subheader("Metric Distributions")
    for col in sect_s.columns:
        fig = plot_hist_kde(sect_s[col].dropna(), col)
        st.pyplot(fig)

    # 7) bar‐chart compare
    st.subheader("Sector vs Global Bars")
    for metric in comp.index:
        keys  = comp.columns.tolist()
        vals  = comp.loc[metric].values
        fig   = hbar_compare(vals, keys, metric, metric)
        st.pyplot(fig)

