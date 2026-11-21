"""Ingredient Reader: the public web app.

Run locally:
    streamlit run app_streamlit.py

The researcher tools are a separate app (researcher_app.py) and are never part
of this deployment (PRD D4).
"""

import streamlit as st

st.set_page_config(
    page_title="Ingredient Reader",
    page_icon=":material/document_scanner:",
    layout="centered",
)

pages = [
    st.Page("ui/pages/scan.py", title="Scan", default=True),
    st.Page("ui/pages/about.py", title="About"),
]
st.navigation(pages, position="top").run()
