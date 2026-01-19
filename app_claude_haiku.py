# Democracia+ Chatbot — Claude-Optimized (RAG with OpenAI Embeddings)
# -----------------------------------------------------------------
# Install:
#   pip install streamlit anthropic openai pypdf numpy
#
# Env vars required:
#   export ANTHROPIC_API_KEY="..."
#   export OPENAI_API_KEY="..."  # embeddings only
#   export DPLUS_ADMIN_PASSWORD="..."  # required for Admin
# Optional:
#   export DPLUS_USER_PASSWORD="..."   # if set, User login requires password
#   export DPLUS_DATA_DIR="data"       # default: data
#
# Run:
#   streamlit run app.py

import os
import json
import time
import uuid
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
CHAT_LOG_PATH = os.path.join(DATA_DIR, "chat_history.jsonl")

DEFAULT_CONFIG: Dict[str, Any] = {
    "chat_model": "claude-3-haiku-20240307",  # ✅ guaranteed access
    "embedding_model": "text-embedding-3-large",
    "temperature": 0.25,
    "top_k": 6,
    "max_history_messages": 8,
    "max_tokens": 1200,
    "default_answer_lang": "auto",
}

# ✅ Updated to stable, widely available model IDs
SUPPORTED_CLAUDE_MODELS = [
    "claude-3-haiku-20240307",  # ✅ only safe model
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


# ----------------------- Basic Helpers -----------------------

def ensure_data_dirs() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)


def check_keys() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Missing env var `ANTHROPIC_API_KEY` (required for Claude chat).")
        st.stop()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing env var `OPENAI_API_KEY` (required for embeddings).")
        st.stop()


def load_config() -> Dict[str, Any]:
    ensure_data_dirs()
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass

    # guard model
    if cfg.get("chat_model") not in SUPPORTED_CLAUDE_MODELS:
        cfg["chat_model"] = DEFAULT_CONFIG["chat_model"]
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    ensure_data_dirs()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def list_document_files() -> List[Tuple[str, str, float]]:
    ensure_data_dirs()
    out: List[Tuple[str, str, float]] = []
    for name in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith((".pdf", ".txt", ".md")):
            continue
        out.append((name, path, os.path.getmtime(path)))
    return out


def delete_document(name: str) -> None:
    path = os.path.join(DOCS_DIR, name)
    if os.path.exists(path):
        os.remove(path)


# ----------------------- Text Parsing & Chunking -----------------------

def read_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts)


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
def build_corpus_from_disk(embed_model: str, file_infos: List[Tuple[str, str, float]]) -> Corpus:
    chunks: List[Chunk] = []
    for name, path, _mtime in file_infos:
        text = load_document_text(path)
        base = os.path.splitext(name)[0]
        for i, chunk_text in enumerate(split_text(text)):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_name=name,
                    section_path=f"{base} (parte {i+1})",
                    page_number=None,
                )
            )

    embeddings = (
        embed_texts_openai(embed_model, [c.text for c in chunks])
        if chunks
        else np.zeros((0, 1), dtype=np.float32)
    )
    return Corpus(chunks=chunks, embeddings=embeddings)


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


# ----------------------- Prompting (Claude) -----------------------

def language_instruction(lang_code: str) -> str:
    if lang_code == "es":
        return "Responde en **español** con tono claro y profesional. Usa párrafos cortos y viñetas cuando ayude."
    if lang_code == "pt":
        return "Responda em **português** com tom claro e profissional. Use parágrafos curtos e marcadores quando ajudar."
    if lang_code == "en":
        return "Answer in **English** with a clear, professional tone. Use short paragraphs and bullets when helpful."
    return "Responde en el mismo idioma del usuario (español o portugués), con tono claro y profesional."


def build_system_prompt(answer_lang: str) -> str:
    return (
        "You are the Democracia+ assistant.\n"
        "You answer using ONLY the provided Democracia+ materials (playbooks, articles, transcripts, documents).\n"
        "If the information is not present in the materials, say so explicitly and do not invent details.\n"
        "When you use the materials, paraphrase and connect ideas to practical guidance for democratic leadership, participation, and institutions.\n"
        "Prefer concrete steps, frameworks, and actionable recommendations.\n\n"
        f"{language_instruction(answer_lang)}"
    )


def format_context(retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return (
            "No relevant excerpts were found in the loaded documents.\n"
            "If you answer, keep it general and clearly state that no specific source material was retrieved."
        )

    parts: List[str] = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        header = f"[Excerpt {i} | {chunk.source_name} | {chunk.section_path} | similarity {score:.3f}]"
        parts.append(f"{header}\n{chunk.text}")
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


def call_claude(
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    messages: List[Dict[str, str]],
) -> str:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    return "".join([block.text for block in resp.content if getattr(block, "type", None) == "text"])


# ----------------------- Chat History Persistence -----------------------

def persist_chat_message(msg: Dict[str, Any]) -> None:
    try:
        ensure_data_dirs()
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": st.session_state.get("session_id"),
            "user_role": st.session_state.get("user_role", "anonymous"),
            "user_name": st.session_state.get("user_name"),
            "message_role": msg.get("role"),
            "content": msg.get("content"),
        }
        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Could not persist chat message: {e}")


def add_message(role: str, content: str) -> None:
    msg = {"role": role, "content": content}
    st.session_state["messages"].append(msg)
    persist_chat_message(msg)


# ----------------------- Auth Helpers -----------------------

def get_admin_password() -> str:
    return os.getenv("DPLUS_ADMIN_PASSWORD", "").strip()


def get_user_password() -> str:
    return os.getenv("DPLUS_USER_PASSWORD", "").strip()


def render_auth_sidebar() -> None:
    st.subheader("Access")

    if st.session_state.get("authenticated"):
        role = st.session_state.get("user_role", "anonymous")
        name = st.session_state.get("user_name")
        label_role = "Admin" if role == "admin" else "User"
        st.success(f"Signed in as {name or label_role}")

        if st.button("Sign out"):
            for key in ["authenticated", "user_role", "user_name", "messages", "session_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    name = st.text_input("Your name", key="login_name")
    requested_role = st.selectbox("Role", ["User", "Admin"], key="login_role")
    password = st.text_input("Access code", type="password", key="login_password")

    if st.button("Sign in"):
        if requested_role == "Admin":
            admin_pw = get_admin_password()
            if not admin_pw:
                st.error("Admin password not configured. Set env var DPLUS_ADMIN_PASSWORD.")
                return
            if password != admin_pw:
                st.error("Incorrect admin access code.")
                return
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = "admin"
        else:
            user_pw = get_user_password()
            if user_pw and password != user_pw:
                st.error("Incorrect access code.")
                return
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = "user"

        st.session_state["user_name"] = name or None
        st.session_state["messages"] = []
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()


# ----------------------- UI: Admin Page -----------------------

def render_admin_page() -> None:
    st.title("D+ Chatbot — Admin")

    cfg = st.session_state["config"]

    st.subheader("Model configuration")

    col1, col2 = st.columns(2)
    with col1:
        cfg["chat_model"] = st.selectbox(
            "Claude model",
            options=SUPPORTED_CLAUDE_MODELS,
            index=SUPPORTED_CLAUDE_MODELS.index(cfg.get("chat_model", DEFAULT_CONFIG["chat_model"])),
        )
        cfg["temperature"] = float(
            st.slider("Temperature", 0.0, 1.0, float(cfg.get("temperature", DEFAULT_CONFIG["temperature"])), 0.05)
        )

    with col2:
        cfg["embedding_model"] = st.text_input(
            "Embedding model (OpenAI)",
            value=cfg.get("embedding_model", DEFAULT_CONFIG["embedding_model"]),
            help="Used only for retrieval (RAG). Claude does not provide embeddings.",
        )
        cfg["top_k"] = int(st.slider("Top K excerpts", 1, 12, int(cfg.get("top_k", DEFAULT_CONFIG["top_k"]))))

    st.subheader("Claude output controls")
    cfg["max_tokens"] = int(st.slider("Max output tokens", 200, 3000, int(cfg.get("max_tokens", DEFAULT_CONFIG["max_tokens"])), 50))
    cfg["max_history_messages"] = int(st.slider("Max chat history turns kept", 0, 20, int(cfg.get("max_history_messages", DEFAULT_CONFIG["max_history_messages"])), 1))

    st.subheader("Default answer language")
    code_to_label = {v: k for k, v in ANSWER_LANG_OPTIONS.items()}
    current_code = cfg.get("default_answer_lang", DEFAULT_CONFIG["default_answer_lang"])
    current_label = code_to_label.get(current_code, "Auto")
    selected_label = st.radio("Language", options=list(ANSWER_LANG_OPTIONS.keys()), index=list(ANSWER_LANG_OPTIONS.keys()).index(current_label))
    cfg["default_answer_lang"] = ANSWER_LANG_OPTIONS[selected_label]

    if st.button("Save configuration"):
        save_config(cfg)
        st.success("Saved.")

    st.divider()
    st.subheader("Documents")

    uploaded = st.file_uploader("Upload (.pdf, .txt, .md)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded:
        ensure_data_dirs()
        for uf in uploaded:
            dest = os.path.join(DOCS_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.read())
        st.success(f"Uploaded {len(uploaded)} file(s).")
        build_corpus_from_disk.clear()

    files = list_document_files()
    if not files:
        st.info("No documents uploaded yet.")
    else:
        st.write(f"**{len(files)} document(s):**")
        for name, _path, mtime in files:
            c1, c2, c3 = st.columns([4, 2, 1])
            with c1:
                st.write(f"📄 {name}")
            with c2:
                st.caption(time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)))
            with c3:
                if st.button("Delete", key=f"del-{name}"):
                    delete_document(name)
                    build_corpus_from_disk.clear()
                    st.warning(f"Deleted {name}.")
                    st.rerun()

    st.divider()
    st.subheader("Index")
    st.write("If you changed docs, rebuilding is automatic (cache clears on upload/delete).")
    if st.button("Force rebuild now"):
        build_corpus_from_disk.clear()
        st.success("Cache cleared. Index will rebuild on next chat request.")


# ----------------------- UI: Chat Page -----------------------

def render_chat_page() -> None:
    st.title("D+ Chatbot — Democracia+")
    st.caption("Claude-powered assistant (Anthropic) with document retrieval (OpenAI embeddings).")

    cfg = st.session_state["config"]
    files = list_document_files()

    if not files:
        st.info("No documents uploaded. Ask an Admin to upload playbooks/articles/transcripts in Admin page.")
        return

    st.sidebar.markdown("### Assistant focus")
    persona = st.sidebar.selectbox(
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

    file_infos = [(n, p, m) for (n, p, m) in files]
    corpus = build_corpus_from_disk(cfg["embedding_model"], file_infos)

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Ask about Democracia+ materials…")
    if not user_input:
        return

    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Retrieving relevant excerpts…"):
        retrieved = retrieve_similar(corpus, user_input, cfg["embedding_model"], int(cfg["top_k"]))

    history = [m for m in st.session_state["messages"] if m["role"] in ("user", "assistant")]
    max_hist = int(cfg.get("max_history_messages", DEFAULT_CONFIG["max_history_messages"]))
    trimmed_history = history[-max_hist:] if max_hist > 0 else []

    user_with_context = build_user_turn_with_context(user_input, retrieved, persona_hint)

    claude_messages: List[Dict[str, str]] = trimmed_history[:-1] if trimmed_history else []
    claude_messages.append({"role": "user", "content": user_with_context})

    system_prompt = build_system_prompt(cfg.get("default_answer_lang", DEFAULT_CONFIG["default_answer_lang"]))

    with st.chat_message("assistant"):
        try:
            answer = call_claude(
                model=cfg["chat_model"],
                temperature=float(cfg["temperature"]),
                max_tokens=int(cfg.get("max_tokens", DEFAULT_CONFIG["max_tokens"])),
                system_prompt=system_prompt,
                messages=claude_messages,
            )
        except Exception as e:
            st.error(f"Claude API error: {e}")
            return

        st.markdown(answer)
        add_message("assistant", answer)

        with st.expander("Sources (excerpts used)"):
            if not retrieved:
                st.write("No relevant excerpts retrieved.")
            else:
                for i, (chunk, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**[{i}]** *{chunk.source_name}* — {chunk.section_path}")
                    st.caption(f"Similarity: {score:.3f}")
                    st.text(chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""))


# ----------------------- Session Init & Main -----------------------

def init_session_state(cfg: Dict[str, Any]) -> None:
    if "config" not in st.session_state:
        st.session_state["config"] = cfg
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = "anonymous"
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = None


def main() -> None:
    st.set_page_config(page_title="D+ Chatbot — Democracia+", page_icon="🗳️", layout="wide")

    ensure_data_dirs()
    check_keys()

    cfg = load_config()
    init_session_state(cfg)

    with st.sidebar:
        st.markdown("## D+ Chatbot")
        render_auth_sidebar()
        if st.session_state.get("authenticated"):
            page = st.radio("Sections", ["Chat", "Admin"], index=0)
        else:
            page = None

    if not st.session_state.get("authenticated"):
        st.info("Please sign in from the sidebar to use the chatbot.")
        return

    if page == "Admin":
        if st.session_state.get("user_role") != "admin":
            st.error("Admins only.")
        else:
            render_admin_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()