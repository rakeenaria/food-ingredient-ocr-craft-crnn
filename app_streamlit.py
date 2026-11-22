"""Ingredient Facts: the public web app.

Run locally:
    streamlit run app_streamlit.py

The researcher tools are a separate app (researcher_app.py) with no route from
this one (PRD D4).
"""

import streamlit as st

from ui.engine import start_loading

st.set_page_config(
    page_title="Ingredient Facts",
    page_icon=":material/document_scanner:",
    layout="centered",
)

# Load the OCR models in the background while the page renders (once per process).
start_loading()

pages = [
    st.Page("ui/pages/scan.py", title="Scan", default=True),
    st.Page("ui/pages/about.py", title="About"),
]
st.navigation(pages, position="top").run()
