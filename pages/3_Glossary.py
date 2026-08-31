"""
Page 4 — Glossary. Plain-language definitions for every column and
metric used across the dashboard.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_common import inject_css, GLOSSARY

st.set_page_config(page_title="CFB Predictor — Glossary", layout="wide", page_icon="📖")
inject_css()


def render():
    st.markdown('<div class="section-label">Reference</div>', unsafe_allow_html=True)
    st.title("Glossary")
    st.caption("What every column and metric in this dashboard actually means.")
    st.divider()

    for term, definition in GLOSSARY:
        st.markdown(f"**{term}**")
        st.markdown(definition)
        st.markdown("")

render()