"""Researcher app: local tools to run, inspect and evaluate checkpoints (PRD R1-R5).

Run with `streamlit run researcher_app.py`. It is never deployed: the public app
(`app_streamlit.py`) has no route to these pages, and they read and write local
files by path.
"""

import streamlit as st

st.set_page_config(page_title="Researcher · Ingredient Reader", page_icon=":material/biotech:", layout="wide")

pages = [
    st.Page("research/pages/run.py", title="Run", default=True),
    st.Page("research/pages/evaluate.py", title="Evaluate"),
    st.Page("research/pages/images.py", title="Edit images"),
]
st.navigation(pages, position="top").run()
