from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .index_store import _embed_texts


def cosine_top_k(embeddings: np.ndarray, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """Return (row_index, cosine_similarity) for top-k rows.

    Defensive against NaN/Inf and zero-norm vectors.
    """
    if embeddings is None or embeddings.size == 0:
        return []

    # Force float32 for stable/fast matmul and avoid overflow from float16/float64 surprises.
    E = np.asarray(embeddings, dtype=np.float32)
    q = np.asarray(query_vec, dtype=np.float32)

    # Clean any NaN/Inf that could come from corrupted loads or upstream issues.
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    # Norms with epsilon guard.
    d_norm = np.linalg.norm(E, axis=1)
    q_norm = float(np.linalg.norm(q))

    if q_norm < 1e-8:
        sims = np.zeros((E.shape[0],), dtype=np.float32)
    else:
        denom = (d_norm * q_norm) + 1e-8
        sims = (E @ q) / denom
        sims = np.where(np.isfinite(sims), sims, -1.0).astype(np.float32)

    k = int(max(1, k))
    k = min(k, sims.shape[0])

    # Faster than full argsort on large corpora.
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def retrieve_sections(
    sections: List[Dict[str, Any]],
    embeddings: np.ndarray,
    query: str,
    embedding_model: str,
    top_k: int,
) -> List[Tuple[Dict[str, Any], float]]:
    """Embed query and return top-k (section, score) using cosine similarity."""
    if not sections or embeddings is None or embeddings.size == 0:
        return []

    q_vec = _embed_texts(embedding_model, [query])[0]
    ranked = cosine_top_k(embeddings, q_vec, int(top_k))
    return [(sections[i], score) for i, score in ranked]
