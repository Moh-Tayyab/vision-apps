"""Streamlit Admin Dashboard for Face Authorization (App 3).

Communicates with the FastAPI backend to manage enrolled persons:
  - Dashboard: stats + recent events
  - Enroll: upload 1 front-facing photo + name → enroll
  - Manage Users: list cards with photos + delete
  - Live Detection: upload image or embed live camera stream

Run:
    streamlit run streamlit_app.py --server.port 8501
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8003")

st.set_page_config(
    page_title="Face Authorization — Admin",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api(path: str, method: str = "GET", **kwargs) -> requests.Response:
    """Make a request to the FastAPI backend with timeout."""
    url = f"{FASTAPI_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=30, **kwargs)
        return resp
    except requests.ConnectionError:
        st.error(f"Cannot connect to FastAPI backend at {FASTAPI_URL}")
        st.stop()
    except requests.Timeout:
        st.error("Backend request timed out")
        st.stop()


def _load_persons() -> list:
    """Fetch enrolled persons from the API."""
    resp = _api("/persons")
    if resp.status_code == 200:
        return resp.json().get("persons", [])
    return []


def _load_events(limit: int = 50) -> list:
    """Fetch recent authorization events."""
    resp = _api(f"/events?limit={limit}")
    if resp.status_code == 200:
        return resp.json().get("events", [])
    return []


def _get_photo_url(name: str) -> str:
    """URL for a person's enrollment photo."""
    return f"{FASTAPI_URL}/persons/{name}/photo"


def _health_check() -> dict:
    """Quick health check."""
    resp = _api("/health")
    if resp.status_code == 200:
        return resp.json()
    return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/user-male-circle.png", width=64)
    st.title("Face Authorization")
    st.caption("Admin Dashboard")

    # Health status
    health = _health_check()
    if health:
        st.success(f"Backend: Online")
        st.metric("Enrolled Persons", health.get("enrolled_persons", 0))
        if health.get("model_loaded"):
            st.caption("Model: Loaded")
        else:
            st.warning("Model: Not loaded yet")
    else:
        st.error("Backend: Offline")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Dashboard", "Enroll User", "Manage Users", "Live Detection"],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.header("Dashboard")

    persons = _load_persons()
    events = _load_events(limit=100)

    # Stats cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Persons", len(persons))
    with col2:
        total_embs = sum(p.get("num_embeddings", 0) for p in persons)
        st.metric("Total Embeddings", total_embs)
    with col3:
        unauth = sum(1 for e in events if e.get("status") == "unauthorized")
        st.metric("Unauthorized Events", unauth)

    # Enrolled persons table
    st.subheader("Enrolled Persons")
    if persons:
        for p in persons:
            cols = st.columns([1, 3, 2, 1])
            with cols[0]:
                photo_url = _get_photo_url(p["name"])
                try:
                    resp = requests.get(photo_url, timeout=5)
                    if resp.status_code == 200:
                        import io
                        img = Image.open(io.BytesIO(resp.content))
                        st.image(img, width=60)
                    else:
                        st.image("https://img.icons8.com/3d-fluency/94/user-male-circle.png", width=60)
                except Exception:
                    st.image("https://img.icons8.com/3d-fluency/94/user-male-circle.png", width=60)
            with cols[1]:
                st.write(f"**{p['name']}**")
            with cols[2]:
                st.caption(f"{p['num_embeddings']} embeddings")
            with cols[3]:
                st.caption("Active")
    else:
        st.info("No persons enrolled yet. Go to **Enroll User** to add someone.")

    # Recent events
    st.subheader("Recent Authorization Events")
    if events:
        import pandas as pd

        df = pd.DataFrame(events[-20:])
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        display_cols = [c for c in ["time", "matched_name", "status", "confidence", "distance"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No events recorded yet.")

# ---------------------------------------------------------------------------
# Page: Enroll User
# ---------------------------------------------------------------------------
elif page == "Enroll User":
    st.header("Enroll New User")
    st.caption("Upload **1 clear front-facing photo** of the person.")

    with st.form("enroll_form", clear_on_submit=True):
        name = st.text_input("Person Name", placeholder="e.g. Ahmed Khan")
        photo = st.file_uploader(
            "Front-facing photo",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, front-facing photo. Well-lit, single face preferred.",
        )

        # Photo preview
        if photo:
            st.divider()
            st.subheader("Photo Preview")
            img = Image.open(photo)
            st.image(img, caption="Enrollment photo", width=300)

        submitted = st.form_submit_button("Enroll", type="primary", use_container_width=True)

    if submitted:
        if not name or not name.strip():
            st.error("Please enter a name.")
        elif photo is None:
            st.error("Please upload a photo.")
        else:
            with st.spinner("Enrolling... (extracting face embedding)"):
                files = {"files": (photo.name, photo.getvalue(), photo.type)}
                data = {"name": name.strip()}
                resp = _api("/persons/enroll", method="POST", data=data, files=files)

            if resp.status_code == 200:
                result = resp.json()
                st.success(
                    f"Enrolled **{result['name']}** — "
                    f"{result['new_embeddings']} embedding(s) saved. "
                    f"Total: {result['total_embeddings']}"
                )
                st.balloons()
            elif resp.status_code == 422:
                st.error(f"Enrollment failed: {resp.json().get('detail', 'Unknown error')}")
            else:
                st.error(f"Server error ({resp.status_code}): {resp.text}")

# ---------------------------------------------------------------------------
# Page: Manage Users
# ---------------------------------------------------------------------------
elif page == "Manage Users":
    st.header("Manage Enrolled Users")

    persons = _load_persons()

    if not persons:
        st.info("No persons enrolled yet.")
        st.stop()

    for p in persons:
        name = p["name"]
        with st.container(border=True):
            cols = st.columns([1, 3, 2, 1])
            with cols[0]:
                photo_url = _get_photo_url(name)
                try:
                    resp = requests.get(photo_url, timeout=5)
                    if resp.status_code == 200:
                        import io
                        img = Image.open(io.BytesIO(resp.content))
                        st.image(img, width=100)
                    else:
                        st.image("https://img.icons8.com/3d-fluency/94/user-male-circle.png", width=100)
                except Exception:
                    st.image("https://img.icons8.com/3d-fluency/94/user-male-circle.png", width=100)
            with cols[1]:
                st.subheader(name)
                st.caption(f"Embeddings: {p['num_embeddings']}")
                if p.get("updated_at"):
                    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(p["updated_at"]))
                    st.caption(f"Enrolled: {ts}")
            with cols[2]:
                pass
            with cols[3]:
                if st.button("Delete", key=f"del_{name}", type="secondary"):
                    resp = _api(f"/persons/{name}", method="DELETE")
                    if resp.status_code == 200:
                        st.success(f"Deleted {name}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to delete: {resp.text}")

# ---------------------------------------------------------------------------
# Page: Live Detection
# ---------------------------------------------------------------------------
elif page == "Live Detection":
    st.header("Live Detection")

    tab1, tab2 = st.tabs(["Upload Image", "Live Camera Stream"])

    # --- Tab: Upload Image ---
    with tab1:
        st.subheader("Verify Image")
        uploaded = st.file_uploader(
            "Upload an image to verify",
            type=["jpg", "jpeg", "png"],
            key="verify_upload",
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width=400)

            if st.button("Verify", type="primary"):
                with st.spinner("Analyzing faces..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = _api("/verify", method="POST", files=files)

                if resp.status_code == 200:
                    result = resp.json()
                    num_faces = result.get("num_faces", 0)
                    st.info(f"Detected **{num_faces}** face(s)")

                    for i, face in enumerate(result.get("faces", [])):
                        status = face.get("status", "unknown")
                        color = {"authorized": "green", "unauthorized": "red", "unknown": "orange"}.get(status, "gray")

                        with st.container(border=True):
                            cols = st.columns([2, 1])
                            with cols[0]:
                                if status == "authorized":
                                    st.success(f"**AUTHORIZED** — {face.get('matched_name', '?')}")
                                elif status == "unauthorized":
                                    st.error(f"**UNAUTHORIZED** — {face.get('matched_name', '?')}")
                                else:
                                    st.warning(f"**UNKNOWN** — No match found")

                                st.caption(f"Confidence: {face.get('confidence', 0):.1%}")
                                if "distance" in face:
                                    st.caption(f"Cosine distance: {face['distance']:.4f}")
                            with cols[1]:
                                st.json(face)
                else:
                    st.error(f"Verification failed: {resp.text}")

    # --- Tab: Live Camera Stream ---
    with tab2:
        st.subheader("Live Camera Feed")
        st.caption("The annotated stream shows: green = authorized, red = unauthorized, orange = unknown.")

        stream_url = f"{FASTAPI_URL}/stream/detect"
        st.markdown(
            f'<iframe src="{stream_url}" width="100%" height="480" frameborder="0"></iframe>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("Alternative: Use the raw stream (no annotations)")
        raw_url = f"{FASTAPI_URL}/stream"
        st.markdown(
            f'<iframe src="{raw_url}" width="100%" height="360" frameborder="0"></iframe>',
            unsafe_allow_html=True,
        )
