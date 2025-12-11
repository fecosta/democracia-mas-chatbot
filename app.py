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
import uuid
from datetime import datetime
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
CHAT_LOG_PATH = os.path.join(DATA_DIR, "chat_history.jsonl")

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


# ----------------------- Utility helpers -----------------------

def check_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "No se encontró la variable de entorno `OPENAI_API_KEY`. "
            "Defínala antes de ejecutar la aplicación."
        )
        st.stop()


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
            # start new buffer with overlap from the end of the previous buf
            if len(p) > max_chars:
                # Hard split big paragraph
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end])
                    start = end - overlap_chars
            else:
                buf = p

    if buf:
        chunks.append(buf)

    # Add overlaps
    final_chunks: List[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            final_chunks.append(c)
        else:
            prev = chunks[i - 1]
            overlap = prev[-overlap_chars:]
            final_chunks.append(overlap + "\n\n" + c)

    return final_chunks


def read_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
    return "\n\n".join(texts)


def load_document_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def embed_texts(model: str, texts: List[str]) -> np.ndarray:
    """
    Call OpenAI embeddings API for a batch of texts.
    """
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vectors)


def build_corpus_from_disk(embed_model: str, file_infos: List[Tuple[str, str, float]]) -> Corpus:
    """
    Build or rebuild the corpus from documents on disk.
    """
    chunks: List[Chunk] = []

    for name, path, mtime in file_infos:
        text = load_document_text(path)
        base_name = os.path.splitext(name)[0]
        for i, chunk_text in enumerate(split_text(text)):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_name=name,
                    section_path=f"{base_name} (parte {i+1})",
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
        return (
            "Responde en **español** con un tono claro, accesible y profesional. "
            "Usa párrafos cortos y, cuando sea útil, listas con viñetas."
        )
    if lang_code == "pt":
        return (
            "Responda em **português** com um tom claro, acessível e profissional. "
            "Use parágrafos curtos e, quando útil, listas com marcadores."
        )
    if lang_code == "en":
        return (
            "Answer in **English** with a clear, accessible and professional tone. "
            "Use short paragraphs and bullet lists when helpful."
        )
    # auto: decide based on user input
    return (
        "Responde en el mismo idioma de la pregunta (español o portugués) "
        "con un tono claro, accesible y profesional."
    )


def build_system_prompt(answer_lang: str) -> str:
    return (
        "Eres el asistente de Democracia+, una organización dedicada a fortalecer "
        "la democracia y el liderazgo político en América Latina.\n\n"
        "Tu tarea es responder preguntas usando exclusivamente los materiales "
        "proporcionados (playbooks, artículos, transcripciones y otros documentos "
        "de Democracia+). Si la información no aparece en los documentos, "
        "sé transparente y di que no tienes datos suficientes.\n\n"
        "Cuando cites ideas específicas de los materiales, intenta parafrasear "
        "y conectar los conceptos con ejemplos prácticos de participación política, "
        "liderazgo y fortalecimiento institucional.\n\n"
        f"{language_instruction(answer_lang)}"
    )


def build_messages(
    user_query: str,
    retrieved_chunks: List[Tuple[Chunk, float]],
    history: List[Dict[str, str]],
    answer_lang: str,
) -> List[Dict[str, str]]:
    """
    Build chat history for the OpenAI Chat API.
    """
    system_prompt = build_system_prompt(answer_lang)

    context_intro = (
        "A continuación tienes fragmentos relevantes de los materiales de Democracia+. "
        "Úsalos como base para tu respuesta. Si algún fragmento no es relevante, "
        "ignóralo.\n\n"
    )

    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        header = f"[Fragmento {i} — {chunk.source_name} — {chunk.section_path} — similitud {score:.3f}]"
        context_blocks.append(f"{header}\n{chunk.text}")

    context_text = context_intro + "\n\n".join(context_blocks) if context_blocks else (
        "No se encontraron fragmentos relevantes en los documentos. "
        "Responde de forma general, aclarando que no estás usando material específico."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": (
                "Contexto documental disponible para esta pregunta:\n\n"
                f"{context_text}"
            ),
        },
    ]

    # history trimming
    max_hist = DEFAULT_CONFIG["max_history_messages"]
    trimmed_history = history[-max_hist:] if max_hist > 0 else []

    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_query})
    return messages


def call_chat_completion(
    model: str,
    temperature: float,
    messages: List[Dict[str, str]],
) -> str:
    client = OpenAI()
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
    # Authentication & identity
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = "anonymous"
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = None


def add_message(role: str, content: str):
    """Append a message to the in-memory chat and persist it to disk.

    Parameters
    ----------
    role:
        "user" or "assistant".
    content:
        Message content.
    """
    msg = {"role": role, "content": content}
    st.session_state["messages"].append(msg)
    persist_chat_message(msg)


def persist_chat_message(msg: Dict[str, Any]) -> None:
    """Safely append a chat message to a JSONL log file.

    This gives us a simple, append-only audit trail of all conversations,
    which can later be analyzed offline or migrated to a proper database
    (for example, Supabase) without changing the rest of the app.
    """
    try:
        # Ensure base folder exists
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
        # Never break the UI because of logging issues
        print(f"[WARN] Could not persist chat message: {e}")


# ----------------------- Authentication helpers -----------------------

def get_admin_password() -> str:
    """Return the administrator access code from environment.

    Set this in your deployment environment as DPLUS_ADMIN_PASSWORD.
    """
    return os.getenv("DPLUS_ADMIN_PASSWORD", "").strip()


def get_user_password() -> str:
    """Return the (optional) regular user access code from environment.

    If DPLUS_USER_PASSWORD is not set, any user can log in as "Usuario"
    without a password. If it is set, the password will be required.
    """
    return os.getenv("DPLUS_USER_PASSWORD", "").strip()


def render_auth_sidebar() -> None:
    """Render the login / logout controls in the sidebar.

    The app distinguishes two roles:
      - "admin": can access the Admin page and manage documents/config.
      - "user": can only use the Chat page.

    Roles are determined by shared access codes configured via environment
    variables (DPLUS_ADMIN_PASSWORD and optionally DPLUS_USER_PASSWORD).
    """
    st.subheader("Acceso")

    # Already logged in
    if st.session_state.get("authenticated"):
        role = st.session_state.get("user_role", "anonymous")
        name = st.session_state.get("user_name")
        label_role = "Admin" if role == "admin" else "Usuario"
        display_name = name or label_role
        st.success(f"Conectado como {display_name}")

        if st.button("Cerrar sesión"):
            # Clear session-related fields and start fresh
            for key in ["authenticated", "user_role", "user_name", "messages", "session_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()
        return

    # Not logged in yet: show login form
    name = st.text_input("Tu nombre", key="login_name")
    requested_role = st.selectbox("Perfil", ["Usuario", "Admin"], key="login_role")
    password = st.text_input("Código de acceso", type="password", key="login_password")

    if st.button("Iniciar sesión"):
        if requested_role == "Admin":
            admin_pw = get_admin_password()
            if not admin_pw:
                st.error(
                    "No hay contraseña de administrador configurada. "
                    "Defina la variable de entorno DPLUS_ADMIN_PASSWORD."
                )
                return
            if password != admin_pw:
                st.error("Código de acceso incorrecto para administrador.")
                return
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = "admin"
        else:
            user_pw = get_user_password()
            if user_pw and password != user_pw:
                st.error("Código de acceso incorrecto.")
                return
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = "user"

        st.session_state["user_name"] = name or None
        # Reset chat for a clean session when logging in
        st.session_state["messages"] = []
        st.session_state["session_id"] = str(uuid.uuid4())
        st.experimental_rerun()


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
            index=max(
                SUPPORTED_CHAT_MODELS.index(cfg.get("chat_model", DEFAULT_CONFIG["chat_model"]))
                if cfg.get("chat_model") in SUPPORTED_CHAT_MODELS
                else 0,
                0,
            ),
        )
        cfg["chat_model"] = chat_model

        temperature = st.slider(
            "Temperatura (creatividad)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=float(cfg.get("temperature", DEFAULT_CONFIG["temperature"])),
        )
        cfg["temperature"] = float(temperature)

    with col2:
        embed_model = st.text_input(
            "Modelo de embeddings",
            value=cfg.get("embedding_model", DEFAULT_CONFIG["embedding_model"]),
            help="Normalmente no es necesario cambiar esto.",
        )
        cfg["embedding_model"] = embed_model

        top_k = st.slider(
            "Número de fragmentos a recuperar (top_k)",
            min_value=1,
            max_value=12,
            value=int(cfg.get("top_k", DEFAULT_CONFIG["top_k"])),
        )
        cfg["top_k"] = int(top_k)

    st.subheader("Idioma de respuesta")
    label_to_code = ANSWER_LANG_OPTIONS
    code_to_label = {v: k for k, v in label_to_code.items()}
    current_code = cfg.get("default_answer_lang", DEFAULT_CONFIG["default_answer_lang"])
    current_label = code_to_label.get(current_code, "Auto")
    selected_label = st.radio(
        "Idioma por defecto",
        options=list(label_to_code.keys()),
        index=list(label_to_code.keys()).index(current_label),
        help="Puede cambiar el idioma de las respuestas del asistente.",
    )
    cfg["default_answer_lang"] = label_to_code[selected_label]

    if st.button("Guardar configuración"):
        save_config(cfg)
        st.success("Configuración guardada.")

    st.divider()
    st.subheader("Gestión de documentos")

    # Upload new docs
    uploaded_files = st.file_uploader(
        "Subir documentos (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        ensure_data_dirs()
        for uf in uploaded_files:
            dest_path = os.path.join(DOCS_DIR, uf.name)
            with open(dest_path, "wb") as f:
                f.write(uf.read())
        st.success(f"Se cargaron {len(uploaded_files)} documento(s).")

    # List current docs
    files = list_document_files()
    if not files:
        st.info("Aún no hay documentos cargados.")
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
    answer_lang = cfg.get("default_answer_lang", DEFAULT_CONFIG["default_answer_lang"])

    # Optional persona selector (just an example, can be extended)
    st.sidebar.markdown("### Modo del asistente")
    persona = st.sidebar.selectbox(
        "Elige el foco de la conversación",
        [
            "General Democracia+",
            "Participación política",
            "Liderazgo y formación",
            "Políticas públicas y diseño institucional",
        ],
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

    # Build messages for OpenAI
    history = [m for m in st.session_state["messages"] if m["role"] in ("user", "assistant")]
    messages = build_messages(
        user_query=user_input,
        retrieved_chunks=retrieved,
        history=history[:-1],  # exclude current user message (already appended)
        answer_lang=answer_lang,
    )

    # Persona hint
    persona_hint = ""
    if persona == "Participación política":
        persona_hint = (
            "Enfócate en participación ciudadana, campañas, partidos políticos y formas "
            "de involucrarse en la vida pública."
        )
    elif persona == "Liderazgo y formación":
        persona_hint = (
            "Enfócate en desarrollo de liderazgos, habilidades blandas, formación de equipos "
            "y procesos pedagógicos."
        )
    elif persona == "Políticas públicas y diseño institucional":
        persona_hint = (
            "Enfócate en diseño de políticas públicas, instituciones democráticas y procesos "
            "de toma de decisión."
        )

    if persona_hint:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Ajusta tu respuesta según el siguiente foco de conversación:\n"
                    f"{persona_hint}"
                ),
            }
        )

    # Call OpenAI
    with st.chat_message("assistant"):
        try:
            answer = call_chat_completion(
                model=cfg["chat_model"],
                temperature=float(cfg.get("temperature", DEFAULT_CONFIG["temperature"])),
                messages=messages,
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
        # Authentication & role selection
        render_auth_sidebar()
        if st.session_state.get("authenticated"):
            page = st.radio("Secciones", ["Chat", "Admin"], index=0)
        else:
            page = None

    if not st.session_state.get("authenticated"):
        st.info("Inicie sesión en la barra lateral para comenzar a usar el chatbot.")
        return

    if page == "Chat":
        render_chat_page()
    elif page == "Admin":
        if st.session_state.get("user_role") != "admin":
            st.error("Solo las personas administradoras pueden acceder a esta sección.")
        else:
            render_admin_page()


if __name__ == "__main__":
    main()