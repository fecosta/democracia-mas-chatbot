# Democracia+ Chatbot — Claude + RAG (OpenAI Embeddings) + SQLite Persistence
# -------------------------------------------------------------------------
# Install:
#   pip install streamlit anthropic openai pypdf numpy
#
# Env vars required:
#   export ANTHROPIC_API_KEY="..."
#   export OPENAI_API_KEY="..."  # embeddings only
#
# Bootstrap admin (ONLY if there are no users yet):
#   export DPLUS_ADMIN_PASSWORD="..."  # used to create the first admin user named "admin"
# Optional:
#   export DPLUS_DATA_DIR="data"       # default: data
#
# Run:
#   streamlit run app.py

import os
import json
import time
import uuid
import sqlite3
import hashlib
import secrets
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from anthropic import Anthropic


# ----------------------- Paths & Defaults -----------------------

DATA_DIR = os.environ.get("DPLUS_DATA_DIR", "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
DB_PATH = os.path.join(DATA_DIR, "app.db")

DEFAULT_CONFIG: Dict[str, Any] = {
    "chat_model": "claude-3-haiku-20240307",  # safest for model access
    "embedding_model": "text-embedding-3-large",
    "temperature": 0.25,
    "top_k": 6,
    "max_history_messages": 10,  # stored in DB, but we still trim what we send to model
    "max_tokens": 1200,
    "default_answer_lang": "auto",  # auto | es | pt | en
}

SUPPORTED_CLAUDE_MODELS = [
    "claude-3-haiku-20240307",
]

ANSWER_LANG_OPTIONS = {
    "Auto": "auto",
    "Español": "es",
    "Português": "pt",
    "English": "en",
}


# ----------------------- Data Structures -----------------------

@dataclass
class Chunk:
    text: str
    source_name: str
    section_path: str
    page_number: Optional[int] = None


@dataclass
class Corpus:
    chunks: List[Chunk]
    embeddings: np.ndarray  # (n_chunks, dim)


# ----------------------- Filesystem / Config -----------------------

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def check_keys() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Missing env var `ANTHROPIC_API_KEY`.")
        st.stop()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing env var `OPENAI_API_KEY` (embeddings).")
        st.stop()


def load_config() -> Dict[str, Any]:
    ensure_dirs()
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass
    if cfg.get("chat_model") not in SUPPORTED_CLAUDE_MODELS:
        cfg["chat_model"] = DEFAULT_CONFIG["chat_model"]
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    ensure_dirs()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ----------------------- SQLite DB -----------------------

def db_connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin','user')),
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        uploaded_by_user_id TEXT,
        uploaded_at TEXT NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(uploaded_by_user_id) REFERENCES users(id)
    );
    """)

    conn.commit()
    conn.close()


# ----------------------- Password hashing -----------------------

def _hash_password(password: str, salt_hex: Optional[str] = None) -> str:
    # PBKDF2-HMAC-SHA256
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, dk_hex = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        candidate = _hash_password(password, salt_hex=salt_hex)
        return secrets.compare_digest(candidate, stored)
    except Exception:
        return False


# ----------------------- User management (DB) -----------------------

def user_count() -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM users;")
    c = int(cur.fetchone()["c"])
    conn.close()
    return c


def bootstrap_admin_if_needed() -> None:
    # Create first admin user if DB is empty
    if user_count() > 0:
        return
    admin_pw = (os.getenv("DPLUS_ADMIN_PASSWORD") or "").strip()
    if not admin_pw:
        # Don’t auto-create without a password
        return
    create_user(username="admin", password=admin_pw, role="admin")


def create_user(username: str, password: str, role: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    uid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO users (id, username, password_hash, role, is_active, created_at) VALUES (?,?,?,?,?,?)",
        (uid, username.strip().lower(), _hash_password(password), role, 1, now),
    )
    conn.commit()
    conn.close()


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    row = cur.fetchone()
    conn.close()
    return row


def list_users() -> List[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY created_at DESC;")
    rows = cur.fetchall()
    conn.close()
    return rows


def set_user_active(user_id: str, is_active: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
    conn.commit()
    conn.close()


def set_user_role(user_id: str, role: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    conn.close()


def reset_user_password(user_id: str, new_password: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), user_id))
    conn.commit()
    conn.close()


# ----------------------- Conversations & Messages (DB) -----------------------

def create_conversation(user_id: str, title: str) -> str:
    conn = db_connect()
    cur = conn.cursor()
    cid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at, is_deleted) VALUES (?,?,?,?,?,0)",
        (cid, user_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return cid


def list_conversations(user_id: str) -> List[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM conversations WHERE user_id = ? AND is_deleted = 0 ORDER BY updated_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def soft_delete_conversation(conversation_id: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET is_deleted = 1 WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def add_message_db(conversation_id: str, role: str, content: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    mid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (mid, conversation_id, role, content, now),
    )
    cur.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    conn.close()


def load_messages(conversation_id: str) -> List[Dict[str, str]]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ----------------------- Documents (disk + DB) -----------------------

def list_documents_db() -> List[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents WHERE is_deleted = 0 ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def store_uploaded_document(filename: str, content: bytes, uploaded_by_user_id: Optional[str]) -> None:
    ensure_dirs()
    safe_name = filename.replace("/", "_").replace("\\", "_")
    digest = sha256_bytes(content)

    # store on disk (dedupe by hash)
    stored_name = f"{digest[:16]}__{safe_name}"
    stored_path = os.path.join(DOCS_DIR, stored_name)

    if not os.path.exists(stored_path):
        with open(stored_path, "wb") as f:
            f.write(content)

    # store in DB
    conn = db_connect()
    cur = conn.cursor()
    doc_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO documents (id, filename, stored_path, sha256, uploaded_by_user_id, uploaded_at, is_deleted) VALUES (?,?,?,?,?,?,0)",
        (doc_id, safe_name, stored_path, digest, uploaded_by_user_id, now),
    )
    conn.commit()
    conn.close()


def delete_document_db(doc_id: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE documents SET is_deleted = 1 WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


def get_active_doc_paths() -> List[Tuple[str, str, float]]:
    # returns tuples (display_name, path, mtime) for building corpus cache key
    docs = list_documents_db()
    out: List[Tuple[str, str, float]] = []
    for d in docs:
        path = d["stored_path"]
        name = d["filename"]
        if os.path.exists(path):
            out.append((name, path, os.path.getmtime(path)))
    return out


# ----------------------- Text parsing / chunking -----------------------

def read_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n\n".join(texts)


def load_document_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_text(text: str, max_chars: int = 3500, overlap_chars: int = 500) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""

    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end])
                    start = max(0, end - overlap_chars)
                buf = ""
            else:
                buf = p

    if buf:
        chunks.append(buf)

    final_chunks: List[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            final_chunks.append(c)
        else:
            overlap = chunks[i - 1][-overlap_chars:]
            final_chunks.append(overlap + "\n\n" + c)

    return final_chunks


# ----------------------- Embeddings (OpenAI) -----------------------

def embed_texts_openai(model: str, texts: List[str]) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vectors)


@st.cache_data(show_spinner=False)
def build_corpus(embed_model: str, doc_infos: List[Tuple[str, str, float]]) -> Corpus:
    chunks: List[Chunk] = []
    for name, path, _mtime in doc_infos:
        text = load_document_text(path)
        base = os.path.splitext(name)[0]
        for i, chunk_text in enumerate(split_text(text)):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_name=name,
                    section_path=f"{base} (part {i+1})",
                )
            )

    if not chunks:
        return Corpus(chunks=[], embeddings=np.zeros((0, 1), dtype=np.float32))

    embs = embed_texts_openai(embed_model, [c.text for c in chunks])
    return Corpus(chunks=chunks, embeddings=embs)


def retrieve_similar(corpus: Corpus, query: str, embed_model: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if not corpus.chunks or corpus.embeddings.size == 0:
        return []

    q_vec = embed_texts_openai(embed_model, [query])[0]
    doc_vecs = corpus.embeddings

    q_norm = np.linalg.norm(q_vec) + 1e-8
    doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-8
    sims = (doc_vecs @ q_vec) / (doc_norms * q_norm)

    k = min(top_k, len(corpus.chunks))
    idxs = np.argsort(-sims)[:k]
    return [(corpus.chunks[i], float(sims[i])) for i in idxs]


# ----------------------- Claude prompting -----------------------

def language_instruction(lang_code: str) -> str:
    if lang_code == "es":
        return "Responde en español, claro y profesional. Usa listas cuando ayuden."
    if lang_code == "pt":
        return "Responda em português, claro e profissional. Use listas quando ajudar."
    if lang_code == "en":
        return "Answer in English, clear and professional. Use bullets when helpful."
    return "Respond in the same language as the user (Spanish or Portuguese), clearly and professionally."


def build_system_prompt(answer_lang: str) -> str:
    return (
        "You are the Democracia+ assistant.\n"
        "Answer using ONLY the provided Democracia+ materials.\n"
        "If the information is not present in the materials, say so and ask for what is missing.\n"
        "Prefer structured, actionable outputs.\n\n"
        f"{language_instruction(answer_lang)}"
    )


def format_context(retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return (
            "No relevant excerpts were found.\n"
            "If you answer, keep it general and explicitly say no specific document excerpt was retrieved."
        )
    parts = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        parts.append(
            f"[Excerpt {i} | {chunk.source_name} | {chunk.section_path} | similarity {score:.3f}]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def build_user_turn_with_context(user_query: str, retrieved: List[Tuple[Chunk, float]], persona_hint: str) -> str:
    ctx = format_context(retrieved)
    persona_block = f"\n\nConversation focus:\n{persona_hint}\n" if persona_hint else ""
    return (
        "Use the following Democracia+ excerpts to answer the question.\n"
        "Rules:\n"
        "- Use only the excerpts as factual basis.\n"
        "- If excerpts are insufficient, say what is missing.\n"
        "- Provide a structured, actionable answer.\n\n"
        f"EXCERPTS:\n{ctx}"
        f"{persona_block}\n\n"
        f"QUESTION:\n{user_query}"
    )


def call_claude(model: str, temperature: float, max_tokens: int, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    return "".join([block.text for block in resp.content if getattr(block, "type", None) == "text"])


# ----------------------- Auth UI (DB-backed) -----------------------

def sign_out() -> None:
    for k in ["auth_user_id", "auth_username", "auth_role", "active_conversation_id"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


def render_login() -> None:
    st.subheader("Sign in")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Sign in"):
        row = get_user_by_username(username)
        if not row or int(row["is_active"]) != 1:
            st.error("Invalid credentials.")
            return
        if not _verify_password(password, row["password_hash"]):
            st.error("Invalid credentials.")
            return

        st.session_state["auth_user_id"] = row["id"]
        st.session_state["auth_username"] = row["username"]
        st.session_state["auth_role"] = row["role"]

        # ensure a conversation exists
        convs = list_conversations(row["id"])
        if convs:
            st.session_state["active_conversation_id"] = convs[0]["id"]
        else:
            st.session_state["active_conversation_id"] = create_conversation(row["id"], "New conversation")
        st.rerun()


# ----------------------- Admin pages -----------------------

def render_admin_users() -> None:
    st.subheader("Users")

    with st.expander("Create user", expanded=False):
        new_u = st.text_input("New username", key="new_username")
        new_p = st.text_input("New password", type="password", key="new_password")
        new_r = st.selectbox("Role", ["user", "admin"], key="new_role")
        if st.button("Create user"):
            try:
                create_user(new_u, new_p, new_r)
                st.success("User created.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not create user: {e}")

    rows = list_users()
    if not rows:
        st.info("No users found.")
        return

    for u in rows:
        st.markdown(f"**{u['username']}** — role: `{u['role']}` — active: `{bool(u['is_active'])}` — created: {u['created_at']}")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            new_role = st.selectbox(
                "Role",
                ["user", "admin"],
                index=0 if u["role"] == "user" else 1,
                key=f"role_{u['id']}",
            )
            if st.button("Save role", key=f"save_role_{u['id']}"):
                set_user_role(u["id"], new_role)
                st.success("Role updated.")
                st.rerun()
        with c2:
            active = st.checkbox("Active", value=bool(u["is_active"]), key=f"active_{u['id']}")
            if st.button("Save active", key=f"save_active_{u['id']}"):
                set_user_active(u["id"], active)
                st.success("Active updated.")
                st.rerun()
        with c3:
            reset_pw = st.text_input("New password", type="password", key=f"resetpw_{u['id']}")
            if st.button("Reset password", key=f"resetpw_btn_{u['id']}"):
                if not reset_pw:
                    st.error("Password required.")
                else:
                    reset_user_password(u["id"], reset_pw)
                    st.success("Password reset.")
                    st.rerun()
        with c4:
            st.caption("Tip: keep at least one admin active.")


def render_admin_documents(current_user_id: str) -> None:
    st.subheader("Documents")

    uploaded = st.file_uploader("Upload (.pdf, .txt, .md)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            store_uploaded_document(uf.name, uf.read(), current_user_id)
        st.success(f"Uploaded {len(uploaded)} file(s).")
        build_corpus.clear()
        st.rerun()

    docs = list_documents_db()
    if not docs:
        st.info("No documents uploaded yet.")
        return

    for d in docs:
        st.markdown(f"📄 **{d['filename']}**  \nsha256: `{d['sha256'][:12]}...`  \nuploaded_at: {d['uploaded_at']}")
        if st.button("Delete", key=f"del_doc_{d['id']}"):
            delete_document_db(d["id"])
            build_corpus.clear()
            st.warning("Document deleted (soft delete).")
            st.rerun()


def render_admin_settings(cfg: Dict[str, Any]) -> None:
    st.subheader("Settings")

    cfg["chat_model"] = st.selectbox("Claude model", SUPPORTED_CLAUDE_MODELS, index=0)
    cfg["temperature"] = float(st.slider("Temperature", 0.0, 1.0, float(cfg["temperature"]), 0.05))
    cfg["max_tokens"] = int(st.slider("Max tokens", 200, 3000, int(cfg["max_tokens"]), 50))
    cfg["top_k"] = int(st.slider("Top K excerpts", 1, 12, int(cfg["top_k"])))

    code_to_label = {v: k for k, v in ANSWER_LANG_OPTIONS.items()}
    current_label = code_to_label.get(cfg.get("default_answer_lang", "auto"), "Auto")
    selected_label = st.radio("Default answer language", list(ANSWER_LANG_OPTIONS.keys()), index=list(ANSWER_LANG_OPTIONS.keys()).index(current_label))
    cfg["default_answer_lang"] = ANSWER_LANG_OPTIONS[selected_label]

    if st.button("Save settings"):
        save_config(cfg)
        st.success("Saved.")
        st.rerun()


# ----------------------- Chat UI -----------------------

def render_chat(cfg: Dict[str, Any], user_id: str) -> None:
    st.title("D+ Chatbot — Democracia+")
    st.caption("Claude (Anthropic) + RAG (OpenAI embeddings) + persistent chat history.")

    # Sidebar: conversation list
    with st.sidebar:
        st.markdown("### Conversations")
        convs = list_conversations(user_id)
        conv_labels = [(c["id"], c["title"]) for c in convs]
        if not conv_labels:
            cid = create_conversation(user_id, "New conversation")
            st.session_state["active_conversation_id"] = cid
            st.rerun()

        active_id = st.session_state.get("active_conversation_id", conv_labels[0][0])
        label_map = {cid: title for cid, title in conv_labels}
        options = [cid for cid, _ in conv_labels]
        selected = st.selectbox(
            "Select conversation",
            options=options,
            format_func=lambda cid: label_map.get(cid, cid),
            index=options.index(active_id) if active_id in options else 0,
        )
        st.session_state["active_conversation_id"] = selected

        c1, c2 = st.columns(2)
        with c1:
            if st.button("New"):
                cid = create_conversation(user_id, "New conversation")
                st.session_state["active_conversation_id"] = cid
                st.rerun()
        with c2:
            if st.button("Delete"):
                soft_delete_conversation(selected)
                st.session_state["active_conversation_id"] = None
                st.rerun()

        st.markdown("---")
        st.markdown("### Focus")
        persona = st.selectbox(
            "Choose a focus",
            [
                "General Democracia+",
                "Citizen participation & political engagement",
                "Leadership & training",
                "Public policy & institutional design",
            ],
            index=0,
        )

    persona_hint = ""
    if persona == "Citizen participation & political engagement":
        persona_hint = "Focus on citizen participation, political organizing, campaigns, parties, and civic engagement."
    elif persona == "Leadership & training":
        persona_hint = "Focus on leadership development, team practices, skills, and training methodologies."
    elif persona == "Public policy & institutional design":
        persona_hint = "Focus on policy design, democratic institutions, governance, and decision-making processes."

    conversation_id = st.session_state["active_conversation_id"]
    msgs = load_messages(conversation_id)

    # Show chat history
    for m in msgs:
        if m["role"] in ("user", "assistant"):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # Build corpus from active docs
    doc_infos = get_active_doc_paths()
    if not doc_infos:
        st.info("No documents uploaded yet. Ask an admin to upload content in Admin area.")
        return

    corpus = build_corpus(cfg["embedding_model"], doc_infos)

    user_input = st.chat_input("Ask about Democracia+ materials…")
    if not user_input:
        return

    # persist user message
    add_message_db(conversation_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Retrieving relevant excerpts…"):
        retrieved = retrieve_similar(corpus, user_input, cfg["embedding_model"], int(cfg["top_k"]))

    # Load conversation history again (including the new user msg)
    msgs = load_messages(conversation_id)
    hist = [m for m in msgs if m["role"] in ("user", "assistant")]
    max_hist = int(cfg.get("max_history_messages", DEFAULT_CONFIG["max_history_messages"]))
    trimmed = hist[-max_hist:] if max_hist > 0 else []

    user_with_context = build_user_turn_with_context(user_input, retrieved, persona_hint)
    claude_messages = trimmed[:-1] if trimmed else []
    claude_messages.append({"role": "user", "content": user_with_context})

    system_prompt = build_system_prompt(cfg.get("default_answer_lang", "auto"))

    with st.chat_message("assistant"):
        try:
            answer = call_claude(
                model=cfg["chat_model"],
                temperature=float(cfg["temperature"]),
                max_tokens=int(cfg["max_tokens"]),
                system_prompt=system_prompt,
                messages=claude_messages,
            )
        except Exception as e:
            st.error(f"Claude API error: {e}")
            return

        st.markdown(answer)
        add_message_db(conversation_id, "assistant", answer)

        with st.expander("Sources (excerpts used)"):
            if not retrieved:
                st.write("No relevant excerpts retrieved.")
            else:
                for i, (chunk, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**[{i}]** *{chunk.source_name}* — {chunk.section_path}")
                    st.caption(f"Similarity: {score:.3f}")
                    st.text(chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""))


# ----------------------- Main -----------------------

def main() -> None:
    st.set_page_config(page_title="D+ Chatbot — Democracia+", page_icon="🗳️", layout="wide")

    ensure_dirs()
    db_init()
    bootstrap_admin_if_needed()
    check_keys()

    cfg = load_config()

    # Sidebar auth + navigation
    with st.sidebar:
        st.markdown("## D+ Chatbot")

        user_id = st.session_state.get("auth_user_id")
        role = st.session_state.get("auth_role")
        username = st.session_state.get("auth_username")

        if not user_id:
            render_login()
            st.info("First run? Set DPLUS_ADMIN_PASSWORD to bootstrap an admin user named `admin`.")
            return

        st.success(f"Signed in as **{username}** (`{role}`)")
        if st.button("Sign out"):
            sign_out()

        st.markdown("---")
        if role == "admin":
            page = st.radio("Sections", ["Chat", "Admin"], index=0)
        else:
            page = "Chat"

    if role != "admin":
        render_chat(cfg, user_id)
        return

    # Admin area
    if page == "Admin":
        st.title("Admin")
        tabs = st.tabs(["Users", "Documents", "Settings"])
        with tabs[0]:
            render_admin_users()
        with tabs[1]:
            render_admin_documents(user_id)
        with tabs[2]:
            render_admin_settings(cfg)
    else:
        render_chat(cfg, user_id)


if __name__ == "__main__":
    main()