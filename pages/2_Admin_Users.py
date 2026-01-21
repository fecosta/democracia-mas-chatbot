import streamlit as st

from core import db
from core.auth import render_sidebar_auth, require_admin, hash_password

st.set_page_config(page_title="Admin — Users", page_icon="👥", layout="wide")

db.init_db()
from core.auth import bootstrap_admin_if_needed
bootstrap_admin_if_needed()

with st.sidebar:
    st.markdown("## D+ Chatbot")
    render_sidebar_auth()

admin = require_admin()

st.title("Admin → Users")

with st.expander("Create user", expanded=False):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["user", "admin"], index=0)
    if st.button("Create"):
        if not username or not password:
            st.error("Username and password required")
        else:
            try:
                db.insert_user(username=username.strip(), password_hash=hash_password(password), role=role)
                st.success("User created")
                st.rerun()
            except Exception as e:
                st.error(f"Could not create user: {e}")

st.markdown("---")

users = db.list_users()
for u in users:
    cols = st.columns([3, 1, 1, 2])
    with cols[0]:
        st.write(f"**{u['username']}**")
        st.caption(f"id: {u['id']} • created: {u['created_at']}")
    with cols[1]:
        st.write(u['role'])
    with cols[2]:
        st.write("✅" if bool(u['is_active']) else "⛔")
    with cols[3]:
        with st.popover("Manage", use_container_width=True):
            new_role = st.selectbox("Role", ["user", "admin"], index=0 if u['role']=="user" else 1, key=f"role_{u['id']}")
            if st.button("Save role", key=f"save_role_{u['id']}"):
                db.set_user_role(u['id'], new_role)
                st.success("Role updated")
                st.rerun()

            active = st.checkbox("Active", value=bool(u['is_active']), key=f"active_{u['id']}")
            if st.button("Save active", key=f"save_active_{u['id']}"):
                db.set_user_active(u['id'], active)
                st.success("Updated")
                st.rerun()

            new_pw = st.text_input("Reset password", type="password", key=f"pw_{u['id']}")
            if st.button("Set password", key=f"set_pw_{u['id']}"):
                if not new_pw:
                    st.error("Password required")
                else:
                    db.set_user_password_hash(u['id'], hash_password(new_pw))
                    st.success("Password updated")

    st.divider()
