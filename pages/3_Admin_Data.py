import os
import shutil
import uuid

import pandas as pd
import streamlit as st

from core import db
from core.auth import render_sidebar_auth, require_admin, bootstrap_admin_if_needed
from core.config import load_config
from core.paths import get_data_dir, docs_dir
from core.utils import ensure_dirs, safe_filename, sha256_bytes
from core.pdf_extract import build_sections_from_pdf, Section
from core.index_store import store_structured_index, clear_index_cache, load_structured_index

user_id = st.session_state.get("auth_user_id")
st.set_page_config(page_title="Admin — Data", page_icon="📄", layout="wide")

db.init_db()
bootstrap_admin_if_needed()

# Sidebar auth
admin = render_sidebar_auth()
require_admin()

cfg = load_config()
DATA_DIR = get_data_dir()
ensure_dirs(DATA_DIR)

st.title("📄 Admin — Data")

# Track reruns to debug upload duplication
st.session_state["admin_data_reruns"] = st.session_state.get("admin_data_reruns", 0) + 1
reruns = st.session_state["admin_data_reruns"]

# -------- Upload (idempotent + logged) --------
st.subheader("Upload & process documents")

with st.form("upload_form", clear_on_submit=True):
    files = st.file_uploader(
        "Select files to upload (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="uploader_files",
    )
    submitted = st.form_submit_button("Upload & Process")

if submitted and files:
    batch_id = str(uuid.uuid4())
    db.log_event(user_id, "upload_submit", details={"batch_id": batch_id, "files_count": len(files), "reruns": reruns})

    uploaded = []
    for uf in files:
        name = safe_filename(uf.name)
        content = uf.getvalue()
        sha = sha256_bytes(content)

        db.log_event(user_id, "upload_seen", filename=name, sha256=sha, details={"batch_id": batch_id, "size": len(content), "reruns": reruns})

        # Dedupe by sha256 (prevents duplicates on Streamlit reruns)
        existing = db.get_document_by_sha256(sha)
        if existing:
            db.log_event(user_id, "upload_skip_duplicate", filename=name, sha256=sha, doc_id=existing["id"], details={"batch_id": batch_id})
            continue

        # Save to disk
        stored_name = f"{sha[:12]}__{name}"
        stored_path = os.path.join(docs_dir(DATA_DIR), stored_name)
        os.makedirs(os.path.dirname(stored_path), exist_ok=True)

        with open(stored_path, "wb") as f:
            f.write(content)

        # Insert into DB
        doc_id = db.insert_document(filename=name, stored_path=stored_path, sha256=sha, structured_dir=None, uploaded_by=user_id)
        db.log_event(user_id, "upload_saved", filename=name, sha256=sha, doc_id=doc_id, details={"batch_id": batch_id, "stored_name": stored_name})

        # Process immediately
        try:
            db.log_event(user_id, "process_start", filename=name, sha256=sha, doc_id=doc_id, details={"batch_id": batch_id})
            if stored_path.lower().endswith(".pdf"):
                sections = build_sections_from_pdf(stored_path, name)
                if not sections:
                    st.warning(f"No text extracted from {name}. It may be scanned or protected.")
                else:
                    sdir = store_structured_index(doc_id, name, sections, cfg["embedding_model"])
                    db.set_document_processed(doc_id, sdir)
            else:
                text = content.decode("utf-8", errors="ignore")
                sections = [Section(path=f"{name}", level=1, page_start=1, page_end=1, text=text)]
                sdir = store_structured_index(doc_id, name, sections, cfg["embedding_model"])
                db.set_document_processed(doc_id, sdir)

            db.log_event(user_id, "process_done", filename=name, sha256=sha, doc_id=doc_id, details={"batch_id": batch_id})
            uploaded.append(name)
        except Exception as e:
            db.log_event(user_id, "process_error", filename=name, sha256=sha, doc_id=doc_id, details={"batch_id": batch_id, "error": str(e)})
            st.error(f"Processing failed for {name}: {e}")

    clear_index_cache()
    st.success(f"Uploaded and processed {len(uploaded)} file(s).")
    st.rerun()

# -------- Documents (table list view) --------
st.markdown("---")
st.subheader("Documents")

docs_rows = db.list_documents(active_only=True)
docs = [dict(r) for r in docs_rows]

if not docs:
    st.info("No documents uploaded yet.")
else:
    # Optional filename filter
    filter_text = st.text_input("Filter by filename", value="", label_visibility="visible")
    if filter_text.strip():
        ft = filter_text.strip().lower()
        docs = [d for d in docs if ft in (d.get("filename") or "").lower()]

    # Build DataFrame for data_editor
    table_rows = []
    for d in docs:
        sha = d.get("sha256") or ""
        table_rows.append({
            "Select": False,
            "id": d.get("id", ""),
            "filename": d.get("filename", ""),
            "uploaded_at": d.get("uploaded_at", ""),
            "processed_at": d.get("processed_at", ""),
            "sha12": (str(sha)[:12] + "…") if sha else "",
            "stored_path": d.get("stored_path", ""),
        })

    df = pd.DataFrame(table_rows)

    # Persist selection across reruns
    if "docs_table_state" not in st.session_state:
        st.session_state["docs_table_state"] = df.copy()

    prev = st.session_state.get("docs_table_state")
    prev_sel = {}
    if prev is not None and not prev.empty and "id" in prev.columns and "Select" in prev.columns:
        prev_sel = dict(zip(prev["id"].astype(str), prev["Select"].astype(bool)))

    df["Select"] = df["id"].astype(str).map(lambda x: bool(prev_sel.get(x, False)))
    st.session_state["docs_table_state"] = df.copy()

    edited = st.data_editor(
        st.session_state["docs_table_state"],
        hide_index=True,
        use_container_width=True,
        disabled=["id", "filename", "uploaded_at", "processed_at", "sha12", "stored_path"],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select"),
            "filename": st.column_config.TextColumn("File"),
            "uploaded_at": st.column_config.TextColumn("Uploaded"),
            "processed_at": st.column_config.TextColumn("Processed"),
            "sha12": st.column_config.TextColumn("SHA"),
            "stored_path": st.column_config.TextColumn("Path"),
        },
        key="docs_table_editor",
    )
    st.session_state["docs_table_state"] = edited

    selected_ids = edited.loc[edited["Select"] == True, "id"].astype(str).tolist()

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Select all"):
            st.session_state["docs_table_state"]["Select"] = True
            st.rerun()
    with c2:
        if st.button("Clear selection"):
            st.session_state["docs_table_state"]["Select"] = False
            st.rerun()
    with c3:
        st.caption(f"Selected: {len(selected_ids)} / {len(edited)}")

    # Delete selected / Delete all
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Delete selected", type="primary", disabled=(len(selected_ids) == 0)):
            for doc_id in selected_ids:
                drow = db.get_document(doc_id)
                if drow:
                    dct = dict(drow)
                    # delete structured dir if exists
                    sdir = dct.get("structured_dir")
                    if sdir and os.path.isdir(sdir):
                        shutil.rmtree(sdir, ignore_errors=True)
                    # optional delete stored file
                    spath = dct.get("stored_path")
                    if spath and os.path.exists(spath):
                        try:
                            os.remove(spath)
                        except Exception:
                            pass

                db.soft_delete_document(doc_id)
                db.log_event(user_id, "delete_doc", doc_id=doc_id, details={"mode": "selected"})
            load_structured_index.clear()
            clear_index_cache()
            st.success(f"Deleted {len(selected_ids)} document(s).")
            del st.session_state["docs_table_state"]
            st.rerun()

    with colB:
        if st.button("Delete ALL documents", type="secondary"):
            st.session_state["confirm_delete_all"] = True

    if st.session_state.get("confirm_delete_all"):
        st.warning("This will delete ALL documents. This cannot be undone.")
        ok, cancel = st.columns([1, 1])
        with ok:
            if st.button("CONFIRM DELETE ALL", type="primary"):
                for d in db.list_documents(active_only=True):
                    dct = dict(d)
                    doc_id = dct.get("id")
                    sdir = dct.get("structured_dir")
                    if sdir and os.path.isdir(sdir):
                        shutil.rmtree(sdir, ignore_errors=True)
                    spath = dct.get("stored_path")
                    if spath and os.path.exists(spath):
                        try:
                            os.remove(spath)
                        except Exception:
                            pass
                    if doc_id:
                        db.soft_delete_document(doc_id)
                        db.log_event(user_id, "delete_doc", doc_id=doc_id, details={"mode": "all"})
                load_structured_index.clear()
                clear_index_cache()
                st.session_state["confirm_delete_all"] = False
                if "docs_table_state" in st.session_state:
                    del st.session_state["docs_table_state"]
                st.success("All documents deleted.")
                st.rerun()
        with cancel:
            if st.button("Cancel"):
                st.session_state["confirm_delete_all"] = False

# -------- Debug events --------
with st.expander("Upload debug / recent events"):
    events = db.list_recent_events(100)
    for e in events:
        ed = dict(e)
        st.code(f"{ed.get('ts')} | {ed.get('action')} | file={ed.get('filename')} | sha={ed.get('sha256')} | doc={ed.get('doc_id')} | {ed.get('details')}")
