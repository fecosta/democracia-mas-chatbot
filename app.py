# Democracia+ Chatbot (Improved Version)
# --------------------------------------
# Requirements (install before running):
#   pip install streamlit openai pypdf numpy
#
# Optional (for deployment / persistence):
#   - Configure a writable "data" folder for documents and index
#   - Set OPENAI_API_KEY env var (and optionally OPENAI_CHAT_MODEL)
#
# Run locally:
#   export OPENAI_API_KEY=your_key_here
#   streamlit run app.py

import os
import io
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ----------------------- Global config & constants -----------------------

DATA_DIR = os.environ.get("DPLUS_DATA_DIR", "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

DEFAULT_CONFIG = {
    "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    "embedding_model": "text-embedding-3-large",
    "temperature": 0.25,
    "top_k": 6,
    "max_history_messages": 6,
    "default_answer_lang": "auto",  # auto | es | pt | en
}

SUPPORTED_CHAT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

ANSWER_LANG_OPTIONS = {
    "Auto": "auto",
    "Español": "es",
    "Português": "pt",
    "English": "en",
}

# ----------------------- Data structures -----------------------

@dataclass
class Chunk:
    text: str
    source_name: str
    section_path: str
    page_number: int | None = None

@dataclass
class Corpus:
    chunks: List[Chunk]
    embeddings: np.ndarray  # shape: (n_chunks, dim)

# ----------------------- OpenAI client -----------------------

client = OpenAI()

def check_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not set. Please add it as an environment variable "
            "in your deployment platform."
        )
        st.stop()

# ----------------------- Persistence helpers -----------------------

def ensure_data_dirs():
    os.makedirs(DOCS_DIR, exist_ok=True)

def load_config() -> Dict[str, Any]:
    ensure_data_dirs()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # merge with defaults to avoid missing keys
            updated = DEFAULT_CONFIG.copy()
            updated.update(cfg)
            return updated
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]) -> None:
    ensure_data_dirs()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def list_document_files() -> List[Tuple[str, str, float]]:
    """
    Returns list of (name, path, mtime) for each document in DOCS_DIR.
    """
    ensure_data_dirs()
    files: List[Tuple[str, str, float]] = []
    for name in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith((".pdf", ".txt", ".md")):
            continue
        mtime = os.path.getmtime(path)
        files.append((name, path, mtime))
    return files

def delete_document(name: str) -> None:
    path = os.path.join(DOCS_DIR, name)
    if os.path.exists(path):
        os.remove(path)

# ----------------------- Text splitting & embeddings -----------------------

def split_text(text: str, max_chars: int = 3500, overlap_chars: int = 500) -> List[str]:
    """
    Simple paragraph splitter with overlap. Works on characters, not tokens,
    but is good enough for our use case and keeps dependencies light.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars - overlap_chars :]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

def embed_texts(model: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)

    res = client.embeddings.create(model=model, input=texts)
    # We assume text-embedding-3-large dim (3072). If another is used, numpy will infer.
    vectors = np.array([d.embedding for d in res.data], dtype=np.float32)
    return vectors

@st.cache_resource(show_spinner=True)
def build_corpus_from_disk(embed_model: str, file_infos: List[Tuple[str, str, float]]) -> Corpus:
    """
    file_infos: list of (name, path, mtime). mtime is used so that index is rebuilt
    when a file is updated.
    """
    chunks: List[Chunk] = []

    for name, path, _ in file_infos:
        ext = name.lower().split(".")[-1]
        with open(path, "rb") as f:
            content_bytes = f.read()

        if ext == "pdf":
            reader = PdfReader(io.BytesIO(content_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for j, t in enumerate(split_text(text)):
                    chunks.append(
                        Chunk(
                            text=t,
                            source_name=name,
                            section_path=f"page-{i+1}/chunk-{j+1}",
                            page_number=i + 1,
                        )
                    )
        elif ext in ("txt", "md"):
            text = content_bytes.decode("utf-8", errors="ignore")
            for j, t in enumerate(split_text(text)):
                chunks.append(
                    Chunk(
                        text=t,
                        source_name=name,
                        section_path=f"chunk-{j+1}",
                        page_number=None,
                    )
                )

    embeddings = embed_texts(embed_model, [c.text for c in chunks])
    return Corpus(chunks=chunks, embeddings=embeddings)

# ----------------------- Retrieval -----------------------

def retrieve_similar(corpus: Corpus, query: str, embed_model: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if not corpus.chunks:
        return []

    q_vec = embed_texts(embed_model, [query])[0]
    doc_vecs = corpus.embeddings

    # cosine similarity
    q_norm = np.linalg.norm(q_vec) + 1e-8
    doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-8
    sims = (doc_vecs @ q_vec) / (doc_norms * q_norm)

    top_k = min(top_k, len(corpus.chunks))
    idxs = np.argsort(-sims)[:top_k]
    return [(corpus.chunks[i], float(sims[i])) for i in idxs]

# ----------------------- Prompt construction -----------------------

def language_instruction(lang_code: str) -> str:
    if lang_code == "es":
        return "Responde en español latinoamericano claro y accesible."
    if lang_code == "pt":
        return "Responda em português brasileiro, de forma clara e acessível."
    if lang_code == "en":
        return "Answer in clear and accessible English."
    return (
        "Responde en el mismo idioma de la pregunta. Si la pregunta mezcla idiomas, "
        "prioriza español o portugués según el contenido."
    )

def build_system_prompt() -> str:
    return (
        "Eres el asistente oficial de Democracia+, una organización que trabaja para "
        "fortalecer la democracia en América Latina. Respondes de manera rigurosa, "
        "didáctica y constructiva, siempre fomentando la participación política, la "
        "ética pública y el fortalecimiento institucional.\n\n"
        "Usa EXCLUSIVAMENTE la información proporcionada en los documentos cargados "
        "como contexto. Si no encuentras la respuesta allí, sé honesto y di que no "
        "tienes información suficiente, sugiriendo cómo la persona podría profundizar "
        "el tema.\n\n"
        "Cuando te refieras explícitamente a partes de los documentos, cita las fuentes "
        "entre corchetes con el formato [1], [2], etc."
    )

def build_context_block(retrieved: List[Tuple[Chunk, float]]) -> str:
    lines = []
    for idx, (chunk, score) in enumerate(retrieved, start=1):
        loc = f"{chunk.source_name} — {chunk.section_path}"
        lines.append(f"[{idx}] {loc}\n{chunk.text}")
    return "\n\n".join(lines)

def call_chat_api(
    model: str,
    temperature: float,
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    answer_lang: str,
    history: List[Dict[str, str]],
) -> str:
    system_prompt = build_system_prompt()
    lang_instr = language_instruction(answer_lang)
    context_block = build_context_block(retrieved)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Instrucción de idioma: {lang_instr}"},
        {
            "role": "system",
            "content": (
                "Contexto proveniente de los playbooks y materiales de Democracia+.\n"
                "Utilízalo como base para tus respuestas, citando los fragmentos relevantes "
                "como [1], [2], etc. cuando corresponda.\n\n"
                f"{context_block}"
            ),
        },
    ]

    # Append short conversation history (without previous context blocks)
    for m in history[-8:]:
        messages.append(m)

    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content

# ----------------------- Streamlit UI helpers -----------------------

def init_session_state(cfg: Dict[str, Any]):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "config" not in st.session_state:
        st.session_state["config"] = cfg
    if "last_index_mtime" not in st.session_state:
        st.session_state["last_index_mtime"] = 0.0

def add_message(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content})

# ----------------------- Admin page -----------------------

def render_admin_page():
    st.title("D+ Chatbot — Admin")

    st.markdown(
        "Configure los modelos de OpenAI, suba nuevos documentos y gestione el índice "
        "que el chatbot utiliza para responder."
    )

    cfg = st.session_state["config"]

    st.subheader("Configuración de modelos")
    col1, col2 = st.columns(2)

    with col1:
        chat_model = st.selectbox(
            "Modelo de chat",
            options=SUPPORTED_CHAT_MODELS,
            index=max(SUPPORTED_CHAT_MODELS.index(cfg.get("chat_model", DEFAULT_CONFIG["chat_model"])) if cfg.get("chat_model") in SUPPORTED_CHAT_MODELS else 0, 0),
        )
        temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=float(cfg.get("temperature", DEFAULT_CONFIG["temperature"])), step=0.05)

    with col2:
        answer_lang_label_to_code = ANSWER_LANG_OPTIONS
        current_lang_code = cfg.get("default_answer_lang", DEFAULT_CONFIG["default_answer_lang"])
        current_label = next((label for label, code in answer_lang_label_to_code.items() if code == current_lang_code), "Auto")
        answer_lang_label = st.selectbox("Idioma por defecto de la respuesta", list(answer_lang_label_to_code.keys()), index=list(answer_lang_label_to_code.keys()).index(current_label))
        top_k = st.slider("Número de fragmentos de contexto (Top K)", min_value=2, max_value=12, value=int(cfg.get("top_k", DEFAULT_CONFIG["top_k"])))

    if st.button("Guardar configuración"):
        cfg["chat_model"] = chat_model
        cfg["temperature"] = float(temperature)
        cfg["top_k"] = int(top_k)
        cfg["default_answer_lang"] = answer_lang_label_to_code[answer_lang_label]
        st.session_state["config"] = cfg
        save_config(cfg)
        st.success("Configuración guardada.")

    st.divider()
    st.subheader("Documentos")

    uploaded_files = st.file_uploader(
        "Subir nuevos documentos (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Estos documentos alimentan el conocimiento del chatbot.",
    )

    if uploaded_files:
        ensure_data_dirs()
        for f in uploaded_files:
            dest = os.path.join(DOCS_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
        st.success("Documentos subidos. Recuerde reconstruir el índice si es necesario.")
        st.button("Actualizar lista de documentos", on_click=lambda: None)

    files = list_document_files()
    if not files:
        st.info("No hay documentos cargados todavía.")
    else:
        st.write(f"**{len(files)} documentos cargados:**")
        for name, path, mtime in files:
            cols = st.columns([4, 2, 1])
            with cols[0]:
                st.write(f"📄 {name}")
            with cols[1]:
                st.caption(time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)))
            with cols[2]:
                if st.button("Eliminar", key=f"del-{name}"):
                    delete_document(name)
                    st.warning(f"Documento '{name}' eliminado. Reconstruya el índice.")
                    st.experimental_rerun()

    st.divider()
    st.subheader("Índice de búsqueda")

    st.markdown(
        "El índice de búsqueda se construye a partir de todos los documentos actuales. "
        "Si agrega o elimina documentos, vuelva a construir el índice."
    )

    if st.button("Reconstruir índice ahora"):
        # Clear cached corpus
        build_corpus_from_disk.clear()
        st.session_state["last_index_mtime"] = time.time()
        st.success("Índice borrado; será reconstruido automáticamente la próxima vez que se use el chatbot.")

# ----------------------- Chat page -----------------------

def render_chat_page():
    st.title("D+ Chatbot — Democracia+")
    st.caption("Asistente para contenidos de Democracia+ basado en modelos de OpenAI.")

    cfg = st.session_state["config"]
    files = list_document_files()

    if not files:
        st.info(
            "Aún no hay documentos cargados. Vaya a la pestaña **Admin** para subir "
            "playbooks, artículos o transcripciones."
        )
        return

    embed_model = cfg["embedding_model"]
    top_k = int(cfg.get("top_k", DEFAULT_CONFIG["top_k"]))
    answer_lang_default = cfg.get("default_answer_lang", DEFAULT_CONFIG["default_answer_lang"])

    # Sidebar options
    with st.sidebar:
        st.header("Opciones de respuesta")
        answer_lang_label = st.selectbox("Idioma de la respuesta", list(ANSWER_LANG_OPTIONS.keys()))
        answer_lang = ANSWER_LANG_OPTIONS[answer_lang_label]

        persona = st.selectbox(
            "Enfoque del asistente",
            ["Facilitador/a", "Formador/a de liderazgos", "Diseñador/a de políticas públicas"],
            index=0,
        )

    # Load / build corpus
    file_infos_for_cache = [(name, path, mtime) for (name, path, mtime) in files]
    corpus = build_corpus_from_disk(embed_model, file_infos_for_cache)

    # Show chat history
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # User input
    user_input = st.chat_input("Haz tu pregunta sobre los contenidos de Democracia+…")
    if not user_input:
        return

    # Append user message
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieval
    with st.spinner("Buscando en los materiales de Democracia+…"):
        retrieved = retrieve_similar(
            corpus=corpus,
            query=user_input,
            embed_model=embed_model,
            top_k=top_k,
        )

    # Build answer
    history = st.session_state["messages"][:-1]  # everything before this question
    effective_lang = answer_lang if answer_lang != "auto" else answer_lang_default
    # Add persona nuance as a small extra instruction in the last user message
    persona_hint = (
        "Responde con el enfoque de un/a "
        f"{persona.lower()}, conectando la respuesta con ejemplos prácticos y "
        "recomendaciones accionables."
    )

    composed_query = user_input + "\n\n" + persona_hint

    with st.chat_message("assistant"):
        try:
            answer = call_chat_api(
                model=cfg.get("chat_model", DEFAULT_CONFIG["chat_model"]),
                temperature=float(cfg.get("temperature", DEFAULT_CONFIG["temperature"])),
                query=composed_query,
                retrieved=retrieved,
                answer_lang=effective_lang,
                history=history,
            )
        except Exception as e:
            st.error(f"Error al llamar a la API de OpenAI: {e}")
            return

        st.markdown(answer)
        add_message("assistant", answer)

        # Show sources
        with st.expander("Ver fuentes utilizadas"):
            if not retrieved:
                st.write("No se encontraron fragmentos relevantes en los documentos.")
            else:
                for idx, (chunk, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**[{idx}]** *{chunk.source_name}* — {chunk.section_path}")
                    st.caption(f"Similitud: {score:.3f}")
                    st.text(chunk.text[:400] + ("…" if len(chunk.text) > 400 else ""))

# ----------------------- Main entrypoint -----------------------

def main():
    st.set_page_config(
        page_title="D+ Chatbot — Democracia+",
        page_icon="🗳️",
        layout="wide",
    )

    check_openai_key()
    cfg = load_config()
    init_session_state(cfg)

    with st.sidebar:
        st.markdown("## D+ Chatbot")
        page = st.radio("Secciones", ["Chat", "Admin"], index=0)

    if page == "Chat":
        render_chat_page()
    else:
        render_admin_page()

if __name__ == "__main__":
    main()