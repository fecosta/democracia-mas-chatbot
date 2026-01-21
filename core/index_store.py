from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
from openai import OpenAI

from .paths import get_data_dir, structured_dir as structured_root
from .utils import ensure_dirs, utc_now_iso
from .pdf_extract import Section


def _embed_texts(model: str, texts: List[str]) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)


def store_structured_index(doc_id: str, filename: str, sections: List[Section], embedding_model: str) -> str:
    """Stores structured sections + embeddings on disk and returns the structured_dir path."""
    data_dir = get_data_dir()
    ensure_dirs(data_dir)
    root = structured_root(data_dir)
    doc_dir = os.path.join(root, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "embedding_model": embedding_model,
        "section_count": len(sections),
        "created_at": utc_now_iso(),
    }
    with open(os.path.join(doc_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # store sections jsonl
    sec_path = os.path.join(doc_dir, "sections.jsonl")
    with open(sec_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sections):
            obj = asdict(s)
            obj["section_id"] = f"{doc_id}::s{i:04d}"
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # embeddings
    texts = [s.text for s in sections]
    embs = _embed_texts(embedding_model, texts) if texts else np.zeros((0, 1), dtype=np.float32)
    np.save(os.path.join(doc_dir, f"embeddings__{embedding_model}.npy"), embs)

    return doc_dir


@st.cache_resource(show_spinner=False)
def load_structured_index(embedding_model: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Loads all structured indexes from disk into memory.

    Returns (sections, embeddings) where sections is a list of dicts containing section metadata and text.
    """
    data_dir = get_data_dir()
    ensure_dirs(data_dir)
    root = structured_root(data_dir)
    if not os.path.exists(root):
        return [], np.zeros((0, 1), dtype=np.float32)

    all_sections: List[Dict[str, Any]] = []
    all_embs: List[np.ndarray] = []

    for doc_id in sorted(os.listdir(root)):
        doc_dir = os.path.join(root, doc_id)
        if not os.path.isdir(doc_dir):
            continue
        sec_file = os.path.join(doc_dir, "sections.jsonl")
        emb_file = os.path.join(doc_dir, f"embeddings__{embedding_model}.npy")
        if not (os.path.exists(sec_file) and os.path.exists(emb_file)):
            continue
        try:
            embs = np.load(emb_file)
        except Exception:
            continue

        secs: List[Dict[str, Any]] = []
        try:
            with open(sec_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    secs.append(json.loads(line))
        except Exception:
            continue

        if len(secs) != len(embs):
            # mismatch, skip to avoid indexing errors
            continue

        all_sections.extend(secs)
        all_embs.append(embs)

    if not all_embs:
        return all_sections, np.zeros((0, 1), dtype=np.float32)
    return all_sections, np.vstack(all_embs)


def clear_index_cache() -> None:
    load_structured_index.clear()
