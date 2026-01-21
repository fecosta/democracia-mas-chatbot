import os
import streamlit as st

from core import db
from core.auth import render_sidebar_auth, require_login
from core.config import load_config
from core.index_store import load_structured_index
from core.retrieval import retrieve_sections
from core.llm import build_system_prompt, build_user_turn, call_claude
from core.utils import utc_now_iso

st.set_page_config(page_title="D+ Chat", page_icon="🗳️", layout="wide")

db.init_db()
from core.auth import bootstrap_admin_if_needed
bootstrap_admin_if_needed()

with st.sidebar:
    st.markdown("## D+ Chatbot")
    render_sidebar_auth()

user = require_login()
cfg = load_config()

st.title("Chat")
st.caption("Claude chat + structured RAG index (built on upload).")

# Conversations sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Conversations")
    if st.button("+ New conversation"):
        cid = db.create_conversation(user["id"], "New chat")
        st.session_state["active_conversation_id"] = cid
        st.rerun()

    convs = db.list_conversations(user["id"])
    if not convs:
        cid = db.create_conversation(user["id"], "New chat")
        st.session_state["active_conversation_id"] = cid
        st.rerun()

    ids = [c["id"] for c in convs]
    labels = [c["title"] for c in convs]

    current = st.session_state.get("active_conversation_id")
    if current not in ids:
        st.session_state["active_conversation_id"] = ids[0]
        current = ids[0]

    idx = ids.index(current)
    sel = st.selectbox("Select", options=list(range(len(ids))), format_func=lambda i: labels[i], index=idx)
    st.session_state["active_conversation_id"] = ids[sel]

    with st.expander("Conversation actions"):
        new_title = st.text_input("Rename", value=labels[sel])
        if st.button("Save title"):
            con = db.connect()
            con.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (new_title or "Chat", __import__('core.utils').utils.utc_now_iso(), ids[sel]))
            con.commit(); con.close()
            st.rerun()
        if st.button("Archive"):
            db.archive_conversation(ids[sel])
            st.session_state.pop("active_conversation_id", None)
            st.rerun()

# Persona hint
with st.sidebar:
    st.markdown("---")
    st.markdown("### Assistant focus")
    persona = st.selectbox(
        "Assistant focus",
        [
            "General Democracia+",
            "Citizen participation & political engagement",
            "Leadership & training",
            "Public policy & institutional design",
        ],
        label_visibility="collapsed",
        index=0,
    )
persona_hint = ""
if persona == "Citizen participation & political engagement":
    persona_hint = "Focus on citizen participation, political organizing, campaigns, parties, and civic engagement."
elif persona == "Leadership & training":
    persona_hint = "Focus on leadership development, team practices, skills, and training methodologies."
elif persona == "Public policy & institutional design":
    persona_hint = "Focus on policy design, democratic institutions, governance, and decision-making processes."

conversation_id = st.session_state.get("active_conversation_id")

# Load messages
msgs = db.get_messages(conversation_id)
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about Democracia+ materials…")
if user_input:
    db.add_message(conversation_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Load structured index (cached)
    docs = db.list_documents(active_only=True)
    if not docs:
        st.error("No documents uploaded yet. Ask an admin to upload content in Admin → Data.")
        st.stop()

    with st.spinner("Searching the knowledge base…"):
        sections, embeddings = load_structured_index(cfg["embedding_model"])
        retrieved = retrieve_sections(sections, embeddings, user_input, cfg["embedding_model"], int(cfg["top_k"]))

    # Build Claude messages from last N turns
    hist = db.get_messages(conversation_id)
    max_hist = int(cfg.get("max_history_messages", 10))
    hist_trim = hist[-max_hist:] if max_hist > 0 else []

    # Replace last user msg with context-augmented user msg
    claude_messages = [{"role": r["role"], "content": r["content"]} for r in hist_trim[:-1]]
    claude_messages.append({"role": "user", "content": build_user_turn(user_input, retrieved, persona_hint)})

    system_prompt = build_system_prompt(cfg.get("default_answer_lang", "auto"))

    with st.chat_message("assistant"):
        try:
            ans = call_claude(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model=cfg["chat_model"],
                temperature=float(cfg["temperature"]),
                max_tokens=int(cfg["max_tokens"]),
                system_prompt=system_prompt,
                messages=claude_messages,
            )
        except Exception as e:
            st.error(f"Claude API error: {e}")
            st.stop()
        st.markdown(ans)
        db.add_message(conversation_id, "assistant", ans)

        with st.expander("Sources"):
            if not retrieved:
                st.write("No excerpts retrieved")
            else:
                for i, (sec, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**[{i}]** {sec.get('path','')}")
                    st.caption(f"Pages {sec.get('page_start')}–{sec.get('page_end')} • similarity {score:.3f}")
                    st.text((sec.get('text','')[:700] + ("…" if len(sec.get('text',''))>700 else "")))