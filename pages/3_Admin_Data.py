
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

def row_get(row, key: str, default=None):
    try:
        return row[key]
    except Exception:
        return default

st.set_page_config(page_title="Admin — Data", page_icon="📄", layout="wide")

db.init_db()
bootstrap_admin_if_needed()

with st.sidebar:
    render_sidebar_auth()

admin = require_admin()
cfg = load_config()

data_dir = get_data_dir()
ensure_dirs(data_dir)
docs_root = docs_dir(data_dir)

st.title("Admin — Data upload & processing")
st.caption("Uploads are processed immediately: PDF → structured sections → embeddings.")

# Rerun counter (helps debug duplicates due to Streamlit reruns)
st.session_state["admin_data_reruns"] = st.session_state.get("admin_data_reruns", 0) + 1
reruns = st.session_state["admin_data_reruns"]

# -------- Upload (idempotent + logged) --------
with st.form("upload_form", clear_on_submit=True):
    files = st.file_uploader(
        "Select files to upload (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="uploader_files",
        label_visibility="visible",
    )
    submitted = st.form_submit_button("Upload & Process")

if submitted and files:
    batch_id = str(uuid.uuid4())
    db.log_event(admin["id"], "upload_submit", details={"batch_id": batch_id, "files_count": len(files), "reruns": reruns})

    processed = 0
    skipped = 0

    for uf in files:
        content = uf.getvalue()
        digest = sha256_bytes(content)
        original_name = uf.name
        name = safe_filename(original_name)

        db.log_event(
            admin["id"],
            "upload_seen",
            filename=original_name,
            sha256=digest,
            details={"batch_id": batch_id, "size": len(content), "reruns": reruns},
        )

        # Dedupe guard (prevents duplicates on reruns)
        existing = db.document_by_sha256(digest)
        if existing:
            skipped += 1
            db.log_event(
                admin["id"],
                "upload_skip_duplicate",
                filename=original_name,
                sha256=digest,
                doc_id=existing["id"],
                details={"batch_id": batch_id},
            )
            continue

        # Store file to disk using hash prefix (prevents duplicate disk files)
        stored_filename = f"{digest[:16]}__{name}"
        stored_path = os.path.join(docs_root, stored_filename)
        os.makedirs(docs_root, exist_ok=True)
        with open(stored_path, "wb") as f:
            f.write(content)

        # Insert document row
        doc_id = db.insert_document(
            filename=name,
            stored_path=stored_path,
            sha256=digest,
            structured_dir=None,
            uploaded_by=admin["id"],
        )
        db.log_event(admin["id"], "upload_saved", filename=original_name, sha256=digest, doc_id=doc_id, details={"batch_id": batch_id})

        # Process: build sections + embeddings on upload
        try:
            db.log_event(admin["id"], "process_start", filename=original_name, sha256=digest, doc_id=doc_id, details={"batch_id": batch_id})

            if name.lower().endswith(".pdf"):
                with st.spinner(f"Processing {name}…"):
                    sections = build_sections_from_pdf(stored_path, name)
                    if not sections:
                        db.log_event(admin["id"], "process_no_text", filename=original_name, sha256=digest, doc_id=doc_id, details={"batch_id": batch_id})
                        st.warning(f"No text extracted from {name}. It may be scanned or protected.")
                    else:
                        sdir = store_structured_index(doc_id, name, sections, cfg["embedding_model"])
                        db.set_document_processed(doc_id, sdir)
            else:
                text = content.decode("utf-8", errors="ignore")
                sections = [Section(path=f"{name}", level=1, page_start=1, page_end=1, text=text)]
                sdir = store_structured_index(doc_id, name, sections, cfg["embedding_model"])
                db.set_document_processed(doc_id, sdir)

            db.log_event(admin["id"], "process_done", filename=original_name, sha256=digest, doc_id=doc_id, details={"batch_id": batch_id})
            processed += 1

        except Exception as e:
            db.log_event(
                admin["id"],
                "process_error",
                filename=original_name,
                sha256=digest,
                doc_id=doc_id,
                details={"batch_id": batch_id, "error": str(e)},
            )
            st.error(f"Processing failed for {name}: {e}")

    clear_index_cache()
    try:
        load_structured_index.clear()
    except Exception:
        pass

    st.success(f"Upload complete. Processed: {processed}, Skipped duplicates: {skipped}.")
    st.rerun()

# -------- Documents list (Option B: table list view) --------
st.markdown("---")
st.subheader("Documents")

docs = db.list_documents(active_only=True)
if not docs:
    st.info("No documents uploaded.")
else:
    rows = []
    for d in docs:
        sha = row_get(d, "sha256", "") or ""
        rows.append({
            "Select": False,
            "id": row_get(d, "id", ""),
            "filename": row_get(d, "filename", ""),
            "uploaded_at": row_get(d, "uploaded_at", ""),
            "sha12": (str(sha)[:12] + "…") if sha else "",
            "path": row_get(d, "stored_path", ""),
            })
    df = pd.DataFrame(rows)

    # Persist selection across reruns
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
            "stored_path": st.column_config.TextColumn("Stored path"),
        },
        key="docs_editor_table",
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

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Delete selected", type="primary", disabled=(len(selected_ids) == 0)):
            for doc_id in selected_ids:
                # soft delete in DB
                drow = next((x for x in docs if x["id"] == doc_id), None)
                db.soft_delete_document(doc_id)

                # remove structured folder (optional)
                if drow:
                    sdir = drow.get("structured_dir")
                    if sdir and os.path.exists(sdir):
                        shutil.rmtree(sdir, ignore_errors=True)

                db.log_event(admin["id"], "delete_doc", doc_id=doc_id, details={"mode": "selected"})

            clear_index_cache()
            try:
                load_structured_index.clear()
            except Exception:
                pass

            st.success(f"Deleted {len(selected_ids)} document(s).")
            st.session_state.pop("docs_table_state", None)
            st.rerun()

    with colB:
        if st.button("Delete ALL documents", type="secondary"):
            st.session_state["confirm_delete_all"] = True

    if st.session_state.get("confirm_delete_all"):
        st.warning("This will delete ALL documents. This cannot be undone.")
        ok, cancel = st.columns([1, 1])
        with ok:
            if st.button("CONFIRM DELETE ALL", type="primary"):
                for d in docs:
                    db.soft_delete_document(d["id"])
                    sdir = d.get("structured_dir")
                    if sdir and os.path.exists(sdir):
                        shutil.rmtree(sdir, ignore_errors=True)
                    db.log_event(admin["id"], "delete_doc", doc_id=d["id"], details={"mode": "all"})

                clear_index_cache()
                try:
                    load_structured_index.clear()
                except Exception:
                    pass

                st.session_state["confirm_delete_all"] = False
                st.session_state.pop("docs_table_state", None)
                st.success("All documents deleted.")
                st.rerun()
        with cancel:
            if st.button("Cancel"):
                st.session_state["confirm_delete_all"] = False

with st.expander("Upload debug / recent events"):
    events = db.list_recent_events(100)
    for e in events:
        st.code(
            f"{e['ts']} | {e['action']} | file={e['filename']} | sha={e['sha256']} | doc={e['doc_id']} | {e['details']}"
        )

with st.expander("Maintenance"):
    if st.button("Clear in-memory index cache"):
        clear_index_cache()
        try:
            load_structured_index.clear()
        except Exception:
            pass
        st.success("Cache cleared.")
