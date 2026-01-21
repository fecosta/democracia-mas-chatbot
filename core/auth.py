from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional, Dict

import streamlit as st

from . import db
from .utils import env


def _pbkdf2(password: str, salt: bytes, iters: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)


def hash_password(password: str) -> str:
    iters = 210_000
    salt = secrets.token_bytes(16)
    dk = _pbkdf2(password, salt, iters)
    return f"pbkdf2_sha256${iters}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_hex, dk_hex = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
        got = _pbkdf2(password, salt, iters)
        return hmac.compare_digest(got, expected)
    except Exception:
        return False


def bootstrap_admin_if_needed() -> None:
    if db.users_count() > 0:
        return
    pw = (env("DPLUS_ADMIN_PASSWORD") or "").strip()
    if not pw:
        return
    username = (env("DPLUS_BOOTSTRAP_ADMIN_USERNAME", "admin") or "admin").strip() or "admin"
    try:
        db.insert_user(username=username, password_hash=hash_password(pw), role="admin")
    except Exception:
        pass


def current_user() -> Optional[Dict[str, str]]:
    if not st.session_state.get("authenticated"):
        return None
    return st.session_state.get("user")


def require_login() -> Dict[str, str]:
    u = current_user()
    if not u:
        st.info("Please sign in to continue.")
        st.stop()
    return u


def require_admin() -> Dict[str, str]:
    u = require_login()
    if u.get("role") != "admin":
        st.error("Admin access required.")
        st.stop()
    return u


def render_sidebar_auth() -> None:
    """Renders login/logout in the sidebar and updates session_state."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state.get("authenticated"):
        u = st.session_state.get("user") or {}
        st.sidebar.success(f"Signed in as {u.get('username','')} ({u.get('role','')})")
        if st.sidebar.button("Sign out"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state.pop("active_conversation_id", None)
            st.rerun()
        return

    st.sidebar.subheader("Sign in")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Sign in"):
        row = db.get_user_by_username(username.strip())
        if not row or not bool(row["is_active"]):
            st.sidebar.error("Invalid credentials.")
            return
        if not verify_password(password, row["password_hash"]):
            st.sidebar.error("Invalid credentials.")
            return
        st.session_state["authenticated"] = True
        st.session_state["user"] = {"id": row["id"], "username": row["username"], "role": row["role"]}
        st.session_state.pop("active_conversation_id", None)
        st.rerun()
