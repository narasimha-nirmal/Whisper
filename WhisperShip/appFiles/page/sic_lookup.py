import streamlit as st

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

