import os
import streamlit as st

from core import db
from core.auth import render_sidebar_auth, bootstrap_admin_if_needed
from core.utils import env

st.set_page_config(page_title="D+ Chatbot", page_icon="🗳️", layout="wide")

db.init_db()
bootstrap_admin_if_needed()

with st.sidebar:
    st.markdown("## D+ Chatbot")
    render_sidebar_auth()
    st.markdown("---")
    st.caption("Pages are in the sidebar (Chat + Admin).")

st.title("D+ Chatbot — Democracia+")

st.markdown(
    """
This app is split into independent sections:

- **Chat** — the user interface
- **Admin → Users** — manage accounts and roles
- **Admin → Data** — upload documents and process PDFs into structured data (on upload)
- **Admin → Model** — model & retrieval settings

To bootstrap the first admin user (only when no users exist yet), set:

- `DPLUS_ADMIN_PASSWORD`

Environment keys required:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

"""
)

missing = []
if not env("ANTHROPIC_API_KEY"):
    missing.append("ANTHROPIC_API_KEY")
if not env("OPENAI_API_KEY"):
    missing.append("OPENAI_API_KEY")

if missing:
    st.warning("Missing env vars: " + ", ".join(missing))
else:
    st.success("Environment looks good.")
