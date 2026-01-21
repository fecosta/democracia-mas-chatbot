import os
import shutil
import streamlit as st

from core import db
from core.auth import render_sidebar_auth, require_admin
from core.config import load_config
from core.paths import get_data_dir, docs_dir
from core.utils import ensure_dirs, safe_filename, sha256_bytes
from core.pdf_extract import build_sections_from_pdf
from core.index_store import store_structured_index, clear_index_cache

st.set_page_config(page_title="Admin — Data", page_icon="📄", layout="wide")

db.init_db()
from core.auth import bootstrap_admin_if_needed
bootstrap_admin_if_needed()

with st.sidebar:
    st.markdown("## D+ Chatbot")
    render_sidebar_auth()

admin = require_admin()
cfg = load_config()

data_dir = get_data_dir()
ensure_dirs(data_dir)

docs_root = docs_dir(data_dir)

st.title("Admin — Data upload & processing")
st.caption("Uploads are processed immediately: PDF → structured sections → embeddings.")

uploaded = st.file_uploader("Upload (.pdf, .txt, .md)", type=["pdf","txt","md"], accept_multiple_files=True)
if uploaded:
    for uf in uploaded:
        name = safe_filename(uf.name)
        content = uf.read()
        digest = sha256_bytes(content)
        stored_name = f"{digest[:16]}__{name}"
        stored_path = os.path.join(docs_root, stored_name)
        if not os.path.exists(stored_path):
            with open(stored_path, "wb") as f:
                f.write(content)

        doc_id = db.insert_document(filename=name, stored_path=stored_path, sha256=digest, structured_dir=None, uploaded_by=admin["id"])

        if name.lower().endswith(".pdf"):
            with st.spinner(f"Processing {name}…"):
                sections = build_sections_from_pdf(stored_path, name)
                if not sections:
                    st.warning(f"No text extracted from {name}. It may be scanned or protected.")
                else:
                    sdir = store_structured_index(doc_id, name, sections, cfg["embedding_model"])
                    db.set_document_processed(doc_id, sdir)
        else:
            # Non-PDF: treat as single section
            text = content.decode("utf-8", errors="ignore")
            from core.pdf_extract import Section
            sections = [Section(path=f"{name}", level=1, page_start=1, page_end=1, text=text)]
            sdir = store_structured_index(doc_id, name, sections, cfg["embedding_model"])
            db.set_document_processed(doc_id, sdir)

    clear_index_cache()
    st.success(f"Uploaded and processed {len(uploaded)} file(s).")
    st.rerun()

st.markdown("---")
st.subheader("Documents")
docs = db.list_documents(active_only=True)
if not docs:
    st.info("No documents uploaded.")
else:
    for d in docs:
        cols = st.columns([4,2,2,1])
        with cols[0]:
            st.write(f"📄 **{d['filename']}**")
            st.caption(d['stored_path'])
        with cols[1]:
            st.caption(f"Uploaded: {d['uploaded_at']}")
            st.caption(f"Processed: {d['processed_at'] or '—'}")
        with cols[2]:
            st.caption(f"SHA: {d['sha256'][:12]}…")
        with cols[3]:
            if st.button("Delete", key=f"del_{d['id']}"):
                # soft delete in DB
                db.soft_delete_document(d['id'])
                # remove structured folder (optional)
                sdir = d['structured_dir']
                if sdir and os.path.exists(sdir):
                    shutil.rmtree(sdir, ignore_errors=True)
                clear_index_cache()
                st.rerun()

with st.expander("Maintenance"):
    if st.button("Clear in-memory index cache"):
        clear_index_cache()
        st.success("Cache cleared.")
