from __future__ import annotations

import sqlite3
import uuid
from typing import Dict, List, Optional

from .paths import db_path, get_data_dir
from .utils import ensure_dirs, utc_now_iso


def connect() -> sqlite3.Connection:
    data_dir = get_data_dir()
    ensure_dirs(data_dir)
    con = sqlite3.connect(db_path(data_dir), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def init_db() -> None:
    con = connect()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','user')),
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            structured_dir TEXT,
            uploaded_by TEXT,
            uploaded_at TEXT NOT NULL,
            processed_at TEXT,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(uploaded_by) REFERENCES users(id)
        );
        
        CREATE TABLE IF NOT EXISTS upload_events (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            user_id TEXT,
            action TEXT NOT NULL,
            filename TEXT,
            sha256 TEXT,
            doc_id TEXT,
            details TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_upload_events_ts ON upload_events(ts);
"""
    )
    con.commit()
    con.close()


# --- users ---

def insert_user(username: str, password_hash: str, role: str) -> str:
    con = connect()
    uid = str(uuid.uuid4())
    con.execute(
        "INSERT INTO users (id, username, password_hash, role, is_active, created_at) VALUES (?,?,?,?,?,?)",
        (uid, username, password_hash, role, 1, utc_now_iso()),
    )
    con.commit()
    con.close()
    return uid


def get_user_by_username(username: str):
    con = connect()
    row = con.execute("SELECT * FROM users WHERE username = ? LIMIT 1", (username,)).fetchone()
    con.close()
    return row


def list_users():
    con = connect()
    rows = con.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    con.close()
    return rows


def set_user_active(user_id: str, active: bool) -> None:
    con = connect()
    con.execute("UPDATE users SET is_active=? WHERE id=?", (1 if active else 0, user_id))
    con.commit()
    con.close()


def set_user_role(user_id: str, role: str) -> None:
    con = connect()
    con.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    con.commit()
    con.close()


def set_user_password_hash(user_id: str, password_hash: str) -> None:
    con = connect()
    con.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, user_id))
    con.commit()
    con.close()


def users_count() -> int:
    con = connect()
    n = int(con.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])
    con.close()
    return n


# --- conversations/messages ---

def list_conversations(user_id: str):
    con = connect()
    rows = con.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE user_id=? AND is_archived=0
        ORDER BY updated_at DESC
        """,
        (user_id,),
    ).fetchall()
    con.close()
    return rows


def create_conversation(user_id: str, title: str = "New conversation") -> str:
    con = connect()
    cid = str(uuid.uuid4())
    now = utc_now_iso()
    con.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at, is_archived) VALUES (?,?,?,?,?,0)",
        (cid, user_id, title or "New conversation", now, now),
    )
    con.commit()
    con.close()
    return cid


def archive_conversation(conversation_id: str) -> None:
    con = connect()
    con.execute("UPDATE conversations SET is_archived=1, updated_at=? WHERE id=?", (utc_now_iso(), conversation_id))
    con.commit()
    con.close()


def get_messages(conversation_id: str):
    con = connect()
    rows = con.execute(
        "SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    con.close()
    return rows


def add_message(conversation_id: str, role: str, content: str) -> None:
    con = connect()
    now = utc_now_iso()
    con.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), conversation_id, role, content, now),
    )
    con.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
    con.commit()
    con.close()


# --- documents ---

def insert_document(filename: str, stored_path: str, sha256: str, structured_dir: Optional[str], uploaded_by: Optional[str]) -> str:
    con = connect()
    did = str(uuid.uuid4())
    con.execute(
        """
        INSERT INTO documents (id, filename, stored_path, sha256, structured_dir, uploaded_by, uploaded_at, processed_at, is_deleted)
        VALUES (?,?,?,?,?,?,?,NULL,0)
        """,
        (did, filename, stored_path, sha256, structured_dir, uploaded_by, utc_now_iso()),
    )
    con.commit()
    con.close()
    return did


def set_document_processed(doc_id: str, structured_dir: str) -> None:
    con = connect()
    con.execute(
        "UPDATE documents SET structured_dir=?, processed_at=? WHERE id=?",
        (structured_dir, utc_now_iso(), doc_id),
    )
    con.commit()
    con.close()


def list_documents(active_only: bool = True):
    con = connect()
    if active_only:
        rows = con.execute("SELECT * FROM documents WHERE is_deleted=0 ORDER BY uploaded_at DESC").fetchall()
    else:
        rows = con.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    con.close()
    return rows


def soft_delete_document(doc_id: str) -> None:
    con = connect()
    con.execute("UPDATE documents SET is_deleted=1 WHERE id=?", (doc_id,))
    con.commit()
    con.close()
import json
from datetime import datetime

def _utc_now() -> str:
    return datetime.utcnow().isoformat()

def log_event(user_id: str | None, action: str, filename: str | None = None, sha256: str | None = None,
              doc_id: str | None = None, details: dict | None = None) -> None:
    con = connect()
    con.execute(
        "INSERT INTO upload_events (id, ts, user_id, action, filename, sha256, doc_id, details) VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), _utc_now(), user_id, action, filename, sha256, doc_id, json.dumps(details or {}, ensure_ascii=False)),
    )
    con.commit()
    con.close()

def list_recent_events(limit: int = 200):
    con = connect()
    rows = con.execute("SELECT * FROM upload_events ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()
    con.close()
    return rows

def get_document_by_sha256(sha256_hex: str):
    con = connect()
    row = con.execute(
        "SELECT * FROM documents WHERE sha256=? AND is_deleted=0 ORDER BY uploaded_at DESC LIMIT 1",
        (sha256_hex,),
    ).fetchone()
    con.close()
    return row

def get_document(doc_id: str):
    con = connect()
    row = con.execute("SELECT * FROM documents WHERE id=? LIMIT 1", (doc_id,)).fetchone()
    con.close()
    return row
