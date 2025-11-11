# Requirements (install before running):
#   pip install streamlit openai pypdf langdetect tiktoken numpy
#   (Optional) pip install faiss-cpu
# Run:
#   export OPENAI_API_KEY=your_key_here
#   streamlit run app.py

import os
import io
import hashlib
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from langdetect import detect
from pypdf import PdfReader
from openai import OpenAI

# --------------- Config ---------------
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # default
MAX_CHUNK_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 120
TOP_K = 6
TEMPERATURE = 0.2

# --------------- Utilities ---------------
client = OpenAI()
try:
    _ = client.models.retrieve(OPENAI_CHAT_MODEL)
except Exception:
    pass

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

@dataclass
class Chunk:
    text: str
    source_name: str
    section_path: str
    page_number: int | None = None
    lang: str | None = None

@dataclass
class Corpus:
    chunks: List[Chunk]
    embeddings: np.ndarray

def split_text(text: str, max_chars: int = 3500, overlap_chars: int = 500) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars - overlap_chars:]
            buf = p
    if buf:
        chunks.append(buf)
    with_overlap = []
    prev_tail = ""
    for c in chunks:
        seg = (prev_tail + "\n\n" + c) if prev_tail else c
        with_overlap.append(seg)
        prev_tail = c[-overlap_chars:]
    return with_overlap

@st.cache_resource(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)
    res = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in res.data], dtype=np.float32)

@st.cache_resource(show_spinner=False)
def build_corpus(files_payload: List[Tuple[str, bytes]]) -> Corpus:
    chunks: List[Chunk] = []
    for name, content in files_payload:
        ext = name.lower().split(".")[-1]
        try:
            if ext == "pdf":
                reader = PdfReader(io.BytesIO(content))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    for j, t in enumerate(split_text(text)):
                        try:
                            lang = detect(t[:4000])
                        except Exception:
                            lang = None
                        chunks.append(Chunk(text=t, source_name=name, section_path=f"page-{i+1}/chunk-{j+1}", page_number=i+1, lang=lang))
            elif ext in ("txt", "md"):
                text = content.decode("utf-8", errors="ignore")
                for j, t in enumerate(split_text(text)):
                    try:
                        lang = detect(t[:4000])
                    except Exception:
                        lang = None
                    chunks.append(Chunk(text=t, source_name=name, section_path=f"chunk-{j+1}", page_number=None, lang=lang))
        except Exception as e:
            st.warning(f"Failed to ingest {name}: {e}")
    emb = embed_texts([c.text for c in chunks])
    return Corpus(chunks=chunks, embeddings=emb)

@st.cache_resource(show_spinner=False)
def embed_query(q: str) -> np.ndarray:
    e = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    return np.array(e.data[0].embedding, dtype=np.float32)

def top_k_similar(query_vec: np.ndarray, matrix: np.ndarray, k: int = 6) -> List[int]:
    if matrix.size == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    return np.argsort(-sims)[:k].tolist()

# --------------- Streamlit UI ---------------
st.set_page_config(page_title="Democracia+ Chatbot POC", page_icon="🗳️", layout="wide")
st.title("🗳️ Democracia+ Chatbot — Streamlit POC")
st.caption("Ask questions grounded in Democracia+ playbooks (PDF/TXT). Upload files or use demo snippets.")

with st.sidebar:
    st.header("Content")
    uploaded_files = st.file_uploader("Upload PDFs or TXT", type=["pdf", "txt", "md"], accept_multiple_files=True)
    demo = st.checkbox("Load demo snippets", value=not uploaded_files)
    st.header("Options")
    persona = st.selectbox("Persona", ["Facilitator", "Policy Maker"], index=0)
    answer_lang = st.selectbox("Answer language", ["Auto", "Español", "Português", "English"], index=0)

payload: List[Tuple[str, bytes]] = []
if uploaded_files:
    for f in uploaded_files:
        payload.append((f.name, f.read()))
elif demo:
    demo_es = "Playbook Democracia+: Módulo de Participación Ciudadana\n\nLa participación efectiva requiere transparencia."
    demo_pt = "Playbook Democracia+: Módulo de Governança\n\nImplemente rituais de alinhamento quinzenais."
    payload = [("demo_es.txt", demo_es.encode()), ("demo_pt.txt", demo_pt.encode())]

key_material = "|".join([f"{n}:{_hash_bytes(b)}" for n, b in payload]) if payload else "empty"
if "_corpus_key" not in st.session_state or st.session_state._corpus_key != key_material:
    st.session_state._corpus_key = key_material
    st.session_state.corpus = build_corpus(payload) if payload else Corpus(chunks=[], embeddings=np.zeros((0, 1)))

corpus: Corpus = st.session_state.corpus
user_q = st.text_input("Question (es/pt/en)", placeholder="¿Cómo implemento mecanismos de escucha activa?")
if st.button("Ask", disabled=not user_q):
    try:
        q_lang = detect(user_q)
    except Exception:
        q_lang = "es"

    q_vec = embed_query(user_q)
    idxs = top_k_similar(q_vec, corpus.embeddings, k=TOP_K)
    retrieved = [corpus.chunks[i] for i in idxs]

    context = "\n\n".join([f"[{i+1}] ({c.source_name} – {c.section_path})\n{c.text}" for i, c in enumerate(retrieved)])
    persona_hint = {
        "Facilitator": "You are a facilitator; give practical steps and exercises.",
        "Policy Maker": "You advise officials; give governance guidance.",
    }[persona]

    system = (
        "You are the Democracia+ assistant. Use only the CONTEXT."
        " Cite sources as [1], [2]. If unsure, say you don't know."
        + persona_hint
    )
    user_prompt = f"QUESTION: {user_q}\n\nCONTEXT:\n{context}"

    with st.spinner("Thinking..."):
        try:
            chat = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            )
            answer = chat.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            st.stop()

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources"):
        for i, c in enumerate(retrieved, start=1):
            st.markdown(f"**[{i}]** *{c.source_name}* — {c.section_path}")
            st.text(c.text[:400] + ("..." if len(c.text) > 400 else ""))

st.markdown("---")
st.caption("Democracia+ Streamlit POC — uses gpt-4o-mini by default.")