import streamlit as st

from core import db
from core.auth import render_sidebar_auth, require_admin
from core.config import load_config, save_config, SUPPORTED_CLAUDE_MODELS, ANSWER_LANG_OPTIONS

st.set_page_config(page_title="Admin — Model", page_icon="⚙️", layout="wide")

db.init_db()
from core.auth import bootstrap_admin_if_needed
bootstrap_admin_if_needed()

with st.sidebar:
    st.markdown("## D+ Chatbot")
    render_sidebar_auth()

admin = require_admin()

cfg = load_config()

st.title("Admin — Model setup")

col1, col2 = st.columns(2)
with col1:
    cfg["chat_model"] = st.selectbox("Claude model", SUPPORTED_CLAUDE_MODELS, index=SUPPORTED_CLAUDE_MODELS.index(cfg.get("chat_model", SUPPORTED_CLAUDE_MODELS[0])))
    cfg["temperature"] = float(st.slider("Temperature", 0.0, 1.0, float(cfg.get("temperature", 0.25)), 0.05))
    cfg["max_tokens"] = int(st.slider("Max output tokens", 200, 3000, int(cfg.get("max_tokens", 1200)), 50))
with col2:
    cfg["embedding_model"] = st.text_input("OpenAI embedding model", value=str(cfg.get("embedding_model", "text-embedding-3-large")))
    cfg["top_k"] = int(st.slider("Top K excerpts", 1, 12, int(cfg.get("top_k", 6))))
    cfg["max_history_messages"] = int(st.slider("History turns kept", 0, 20, int(cfg.get("max_history_messages", 10))))

# Language option
code_to_label = {v: k for k, v in ANSWER_LANG_OPTIONS.items()}
current_label = code_to_label.get(cfg.get("default_answer_lang", "auto"), "Auto")
selected_label = st.radio("Default answer language", list(ANSWER_LANG_OPTIONS.keys()), index=list(ANSWER_LANG_OPTIONS.keys()).index(current_label))
cfg["default_answer_lang"] = ANSWER_LANG_OPTIONS[selected_label]

st.markdown("---")
if st.button("Save settings"):
    save_config(cfg)
    st.success("Saved.")
