import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_raw():
    """
    Read and merge the two CSVs into one DataFrame without filtering.
    """
    base = os.path.dirname(__file__)
    itac = pd.read_csv(os.path.join(base, '..', 'itac_compact.csv')) \
             .dropna(subset=['SIC'])
    sic  = pd.read_csv(os.path.join(base, '..', 'sic-codes.csv'))

    itac['SIC'] = itac['SIC'].astype(int)
    sic['SIC']  = sic['SIC'].astype(int)

    return itac.merge(
        sic[['SIC','Description','Major Group']],
        on='SIC',
        how='left'
    )

def filter_ca(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the CA + numeric filters."""
    return df[
        (df.FY >= 2018) &
        (df.STATE == 'CA') &
        (df.ANNUAL_ELECTRICITY_COST.between(100_000, 3_000_000)) &
        (df.TOTAL_RECC_SAVINGS >= 100_000) &
        (df.ROI_YRS <= 5)
    ].reset_index(drop=True)

def filter_national(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only the numeric filters, no STATE restriction."""
    return df[
        (df.FY >= 2018) &
        (df.ANNUAL_ELECTRICITY_COST.between(100_000, 3_000_000)) &
        (df.TOTAL_RECC_SAVINGS >= 100_000) &
        (df.ROI_YRS <= 5)
    ].reset_index(drop=True)

# Metric name → (pretty label, scale factor)
METRICS = {
    'ANNUAL_SALES':            ('Annual Sales (M)',        1e6),
    'EMPLOYEES':               ('Employees',               1),
    'PLANT_AREA':              ('Plant Area (M sqft)',     1e6),
    'ANNUAL_ELECTRICITY_COST': ('Electricity Cost (M)',    1e6),
    'TOTAL_RECC_IMP_COST':     ('Implementation Cost (M)', 1e6),
    'TOTAL_RECC_SAVINGS':      ('Savings (M)',             1e6),
    'ROI_YRS':                 ('ROI (yrs)',               1),
}

def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns in METRICS by applying the scale factor and renaming.
    """
    from .data import METRICS
    return pd.DataFrame({
        label: df[col] / scale
        for col, (label, scale) in METRICS.items()
        if col in df
    })

