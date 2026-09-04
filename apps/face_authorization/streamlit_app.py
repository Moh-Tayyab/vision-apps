"""Streamlit Admin Dashboard for Face Authorization (App 3).

Communicates with the FastAPI backend to manage enrolled persons:
  - Dashboard: stats + recent events
  - Enroll: upload 1 front-facing photo + name → enroll
  - Manage Users: list cards with photos + delete
  - Live Detection: Multi-source Live Stream (USB / Mobile IP / Browser), In-browser Camera, and Image Upload

Run:
    streamlit run streamlit_app.py --server.port 8501
"""

from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8003")
API_KEY = os.getenv("FACE_AUTH_API_KEY", "face-auth-dev-key-2026")

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
    """Make a request to the FastAPI backend with timeout and auth headers."""
    url = f"{FASTAPI_URL}{path}"
    headers = kwargs.pop("headers", {})
    if "X-API-Key" not in headers:
        headers["X-API-Key"] = API_KEY
    try:
        resp = requests.request(method, url, timeout=30, headers=headers, **kwargs)
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


def _load_audit_events_db(limit: int = 50, offset: int = 0, status: str = None, search_name: str = None) -> tuple:
    """Fetch persistent audit logs from database."""
    params = f"?limit={limit}&offset={offset}"
    if status and status.lower() != "all":
        params += f"&status={status.lower()}"
    if search_name and search_name.strip():
        params += f"&name={search_name.strip()}"
    resp = _api(f"/audit/events{params}")
    if resp.status_code == 200:
        d = resp.json()
        return d.get("total", 0), d.get("events", [])
    return 0, []


def _get_db_stats() -> dict:
    """Fetch database & Qdrant vector engine statistics."""
    resp = _api("/db/stats")
    if resp.status_code == 200:
        return resp.json()
    return {}


def _get_photo_url(name: str) -> str:
    """URL for a person's enrollment photo."""
    return f"{FASTAPI_URL}/persons/{name}/photo"


def _health_check() -> dict:
    """Quick health check."""
    resp = _api("/health")
    if resp.status_code == 200:
        return resp.json()
    return {}


def _get_camera_devices() -> list:
    """List available USB / V4L2 video devices on the system."""
    resp = _api("/camera/devices")
    if resp.status_code == 200:
        return resp.json().get("devices", [])
    return []


def _get_camera_health() -> dict:
    """Get active camera health."""
    resp = _api("/camera/health")
    if resp.status_code == 200:
        return resp.json()
    return {}


def _get_sensitivity_settings() -> dict:
    """Get current sensitivity and distance settings."""
    resp = _api("/settings/sensitivity")
    if resp.status_code == 200:
        return resp.json()
    return {
        "min_face_size": 14,
        "detection_confidence": 0.22,
        "cosine_match_threshold": 0.48,
        "inference_max_width": 720,
    }


def _update_sensitivity_settings(**kwargs) -> bool:
    """Update sensitivity settings."""
    resp = _api("/settings/sensitivity", method="POST", data=kwargs)
    return resp.status_code == 200


ORIENTATION_TRANSFORMS = ["none", "flip_h", "flip_v", "rotate_90_cw", "rotate_90_ccw", "rotate_180"]


def _apply_orientation_change():
    """Auto-apply the orientation selectbox value (no extra button)."""
    val = st.session_state.get("cam_orientation", "none")
    r = _api("/camera/transform", method="POST", data={"transform": val})
    if r.status_code == 200:
        st.toast(f"Orientation → {val}")
    else:
        st.error(f"Orientation failed: {r.text}")


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
        st.success("Backend: Online")
        st.metric("Enrolled Persons", health.get("enrolled_persons", 0))
        if health.get("model_loaded"):
            st.caption("Model: Loaded (Facenet)")
        else:
            st.warning("Model: Initializing...")

        cam_info = health.get("camera", {})
        cam_status = cam_info.get("status", "disconnected")
        if cam_status == "connected":
            st.success(f"Camera: Connected ({cam_info.get('fps', 0)} FPS)")
        elif cam_status == "reconnecting":
            st.warning("Camera: Connecting...")
        else:
            st.info(f"Camera: {cam_status.capitalize()}")

        sens = _get_sensitivity_settings()
        st.caption(f"🎯 Sensitivity: Threshold `{sens.get('cosine_match_threshold', 0.48)}` | Min Face `{sens.get('min_face_size', 14)}px`")
    else:
        st.error("Backend: Offline")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Dashboard", "Enroll User", "Manage Users", "Live Detection", "Audit Logs & Security"],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.header("Dashboard")

    persons = _load_persons()
    db_stats = _get_db_stats()

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Persons", len(persons))
    with col2:
        st.metric("Total Embeddings", db_stats.get("total_embeddings", 0))
    with col3:
        st.metric("Access Violations", db_stats.get("total_violations", 0))
    with col4:
        st.metric("Spoof Attacks Blocked", db_stats.get("total_spoofs_detected", 0))

    # Vector Engine Status Banner
    q_active = db_stats.get("qdrant_active", False)
    if q_active:
        st.success(f"🚀 **Vector Search Engine: Qdrant Active** (HNSW Cosine Index at `{db_stats.get('qdrant_url')}`)")
    else:
        st.info("📦 **Vector Search Engine: SQLite WAL + NumPy Embedded Active** (Zero-latency local index)")

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
        name = st.text_input("Person Name", placeholder="e.g. Muhammad Tayyab")
        photo = st.file_uploader(
            "Front-facing photo",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, front-facing photo. Well-lit, single face preferred.",
        )

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
    st.header("Live Face Authorization")

    tab1, tab2, tab3 = st.tabs(["Live Camera Stream", "In-Browser Webcam", "Upload Image"])

    # --- Tab 1: Live Camera Stream ---
    with tab1:
        st.subheader("Live Video Stream & Camera Controls")

        # Camera Configuration Controls
        with st.expander("⚙️ Camera Settings & Connections", expanded=True):
            cam_health = _get_camera_health()
            devices = _get_camera_devices()
            health_info = _health_check()
            local_ip = health_info.get("local_ip", "192.168.1.39")

            source_mode = st.radio(
                "Camera Source",
                [
                    "📱 Mobile Browser (/mobile)",
                    "📷 Mobile IP Webcam (URL e.g. http://phone-ip:8080/video)",
                    "🔌 USB Webcam Device",
                ],
                index=0 if cam_health.get("source_type") == "mobile" else (1 if cam_health.get("source_type") == "http_mjpeg" else 2),
                horizontal=True,
            )

            if source_mode.startswith("📱"):
                st.info(
                    f"📲 **Phone Browser Instructions:**\n\n"
                    f"1. Open **`https://{local_ip}:8445/mobile`** on your phone (tap *Advanced → Proceed* for SSL).\n"
                    f"2. OR open **`http://{local_ip}:8003/mobile`** and tap **'📸 Snap Photo & Send'** (works without SSL).\n"
                    f"3. Tap **Start Stream** to send live camera frames directly."
                )
                selected_uri, st_type = "browser", "mobile"
            elif source_mode.startswith("📷"):
                st.info(
                    "📱 **Android IP Webcam App Instructions:**\n\n"
                    "1. Open the **IP Webcam** app on your Android phone.\n"
                    "2. Scroll to the bottom and tap **'Start server'**.\n"
                    "3. Look at the IP displayed on your phone's screen (e.g. `http://192.168.1.XX:8080`).\n"
                    "4. Enter `http://<PHONE_IP>:8080/video` in the text box below and click **▶ Connect**.\n"
                    "*(Note: 192.168.1.39 is the laptop IP; your phone has its own IP assigned by the Wi-Fi router)*"
                )
                col_u1, col_u2 = st.columns([3, 1])
                with col_u1:
                    default_val = cam_health.get("source_uri") if (cam_health.get("source_type") == "http_mjpeg" and cam_health.get("source_uri")) else ""
                    selected_uri = st.text_input(
                        "Mobile Phone IP Webcam Stream URL",
                        value=default_val,
                        placeholder="http://192.168.1.XX:8080/video",
                        help="Enter the URL shown on your phone screen e.g. http://192.168.1.45:8080/video",
                    )
                with col_u2:
                    st.write("")
                    st.write("")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("▶ Connect", type="primary", use_container_width=True):
                            if not selected_uri:
                                st.warning("Please enter your phone's IP Webcam URL first.")
                            else:
                                with st.spinner("Connecting IP Camera..."):
                                    resp = _api(
                                        "/camera/connect",
                                        method="POST",
                                        data={"source_uri": selected_uri, "fps": 15},
                                    )
                                if resp.status_code == 200:
                                    st.success("Connected!")
                                    time.sleep(0.4)
                                    st.rerun()
                                else:
                                    st.error(f"Connect failed: {resp.text}")
                    with col_b2:
                        if st.button("⏹ Disconnect", use_container_width=True):
                            _api("/camera/disconnect", method="POST")
                            st.rerun()
                st_type = "http_mjpeg"
            else:
                dev_dict = {
                    f"{d['id']} {'[Laptop Internal Webcam]' if d.get('internal') else '[External USB Cam]'}": d["id"]
                    for d in devices if d.get("available")
                }
                if not dev_dict:
                    dev_dict = {"/dev/video0 [Laptop Internal Webcam]": "/dev/video0"}
                c_usb1, c_usb2 = st.columns([3, 1])
                with c_usb1:
                    selected_label = st.selectbox("USB Video Device", list(dev_dict.keys()))
                    selected_uri = dev_dict[selected_label]
                with c_usb2:
                    st.write("")
                    st.write("")
                    col_u_start, col_u_stop = st.columns(2)
                    with col_u_start:
                        if st.button("▶ Start USB", type="primary", use_container_width=True):
                            dev_idx = 0
                            try:
                                dev_idx = int(selected_uri.replace("/dev/video", ""))
                            except Exception:
                                pass
                            _api(f"/usb/start?device_index={dev_idx}&fps=20", method="POST")
                            st.success("USB Camera started!")
                            time.sleep(0.4)
                            st.rerun()
                    with col_u_stop:
                        if st.button("⏹ Stop USB", use_container_width=True):
                            _api("/usb/stop", method="POST")
                            st.rerun()
                st_type = "usb"

            # Status KPI row
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Status", cam_health.get("status", "Standby").upper())
            with k2:
                st.metric("Stream FPS", f"{cam_health.get('fps', 0):.1f}")
            with k3:
                st.metric("Total Frames", cam_health.get("frame_count", 0))
            with k4:
                st.metric("Connection", cam_health.get("connection_medium", "N/A"))

        # Orientation quick-adjust buttons
        col_rot_label, r1, r2, r3, r4, r5 = st.columns([2, 1, 1, 1, 1, 1])
        with col_rot_label:
            st.caption("🔄 **Fix View Orientation:**")
        with r1:
            if st.button("Normal", use_container_width=True):
                _api("/camera/transform", method="POST", data={"transform": "none"})
                st.rerun()
        with r2:
            if st.button("90° CW", use_container_width=True):
                _api("/camera/transform", method="POST", data={"transform": "rotate_90_cw"})
                st.rerun()
        with r3:
            if st.button("90° CCW", use_container_width=True):
                _api("/camera/transform", method="POST", data={"transform": "rotate_90_ccw"})
                st.rerun()
        with r4:
            if st.button("180°", use_container_width=True):
                _api("/camera/transform", method="POST", data={"transform": "rotate_180"})
                st.rerun()
        with r5:
            if st.button("🪞 Flip", use_container_width=True):
                _api("/camera/transform", method="POST", data={"transform": "flip_h"})
                st.rerun()

        # Live Annotated MJPEG Stream
        st.caption("Live Feed: 🟩 Green = AUTHORIZED, 🟥 Red = UNAUTHORIZED, 🟧 Orange = UNKNOWN")
        
        st.markdown(
            f"""
            <div style="background:#0f172a; border-radius:14px; overflow:hidden; border:2px solid #334155; text-align:center; padding:0; min-height:460px;">
                <iframe src="http://localhost:8003/stream/detect" width="100%" height="460" frameborder="0" style="border:none; display:block; margin:0 auto; background:#0f172a; border-radius:12px;" scrolling="no"></iframe>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.85rem;">
                <span style="color:#94a3b8;">⚡ Ultra-fast 30 FPS Stream Active</span>
                <a href="http://localhost:8003" target="_blank" style="color:#38bdf8; font-weight:600; text-decoration:none;">
                    🖥️ Open Standalone Live Monitor (Port 8003) &rarr;
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sensitivity & Distance Controls
        with st.expander("🎯 **Sensitivity & Distance Tuning (Door Khare Chehron ke Liye)**", expanded=False):
            curr_sens = _get_sensitivity_settings()
            st.caption("Adjust sensitivity to recognize faces from greater distances without standing close to camera.")
            
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                if st.button("🔭 **Long-Distance Mode**\n\n(High Sensitivity)", use_container_width=True):
                    _update_sensitivity_settings(preset="long_distance")
                    st.toast("Preset set to Long-Distance Mode (High Sensitivity)!")
                    time.sleep(0.3)
                    st.rerun()
            with p_col2:
                if st.button("⚖️ **Balanced Mode**\n\n(Standard / Default)", use_container_width=True):
                    _update_sensitivity_settings(preset="balanced")
                    st.toast("Preset set to Balanced Mode!")
                    time.sleep(0.3)
                    st.rerun()
            with p_col3:
                if st.button("🔒 **Strict Mode**\n\n(Close-up / High Security)", use_container_width=True):
                    _update_sensitivity_settings(preset="strict")
                    st.toast("Preset set to Strict Mode!")
                    time.sleep(0.3)
                    st.rerun()

            st.markdown("---")
            st.markdown("##### Fine-Tune Parameters:")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                new_thresh = st.slider(
                    "Cosine Match Threshold (Higher = more lenient)",
                    min_value=0.25,
                    max_value=0.65,
                    value=float(curr_sens.get("cosine_match_threshold", 0.48)),
                    step=0.01,
                    help="Cosine distance limit. 0.48-0.52 allows faces to match from a distance even if slightly lower res.",
                )
            with sc2:
                new_min_size = st.slider(
                    "Min Face Size (px) (Lower = detects far faces)",
                    min_value=10,
                    max_value=60,
                    value=int(curr_sens.get("min_face_size", 14)),
                    step=1,
                    help="Minimum pixel dimension of face bounding box. 14px allows detection across the room.",
                )
            with sc3:
                new_conf = st.slider(
                    "Detection Confidence (Lower = catches distant faces)",
                    min_value=0.10,
                    max_value=0.70,
                    value=float(curr_sens.get("detection_confidence", 0.22)),
                    step=0.01,
                    help="RetinaFace confidence filter. 0.20-0.25 catches smaller faces reliably.",
                )

            if st.button("💾 Apply Custom Sensitivity", type="primary"):
                ok = _update_sensitivity_settings(
                    cosine_match_threshold=new_thresh,
                    min_face_size=new_min_size,
                    detection_confidence=new_conf,
                )
                if ok:
                    st.success("Sensitivity settings updated successfully!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("Failed to update sensitivity settings.")

    # --- Tab 2: In-Browser Webcam Capture ---
    with tab2:
        st.subheader("In-Browser Webcam Quick Check")
        st.caption("Capture a quick snapshot directly from your browser's webcam.")
        camera_photo = st.camera_input("Take a photo")

        if camera_photo is not None:
            with st.spinner("Verifying face..."):
                files = {"file": ("webcam.jpg", camera_photo.getvalue(), "image/jpeg")}
                resp = _api("/verify", method="POST", files=files)

            if resp.status_code == 200:
                result = resp.json()
                num_faces = result.get("num_faces", 0)
                st.info(f"Detected **{num_faces}** face(s)")

                for face in result.get("faces", []):
                    status = face.get("status", "unknown")
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

    # --- Tab 3: Upload Image ---
    with tab3:
        st.subheader("Verify Static Image")
        uploaded = st.file_uploader(
            "Upload an image to verify",
            type=["jpg", "jpeg", "png"],
            key="verify_upload",
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width=400)

            if st.button("Verify Face", type="primary"):
                with st.spinner("Analyzing faces..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = _api("/verify", method="POST", files=files)

                if resp.status_code == 200:
                    result = resp.json()
                    num_faces = result.get("num_faces", 0)
                    st.info(f"Detected **{num_faces}** face(s)")

                    for face in result.get("faces", []):
                        status = face.get("status", "unknown")
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

# ---------------------------------------------------------------------------
# Page: Audit Logs & Security
# ---------------------------------------------------------------------------
elif page == "Audit Logs & Security":
    st.header("Security & Audit Logs")
    st.caption("Persistent tamper-proof database access logs and vector search monitoring.")

    db_stats = _get_db_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Logged Events", db_stats.get("total_audit_events", 0))
    with c2:
        st.metric("Violations Logged", db_stats.get("total_violations", 0))
    with c3:
        st.metric("Spoofing Attempts", db_stats.get("total_spoofs_detected", 0))
    with c4:
        st.metric("Vector Store", "Qdrant HNSW" if db_stats.get("qdrant_active") else "SQLite+NumPy")

    st.divider()

    # Filter Controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        filter_status = st.selectbox("Filter Status", ["All", "authorized", "unauthorized", "spoof"])
    with col_f2:
        search_name = st.text_input("Search by Name", placeholder="e.g. Tayyab")
    with col_f3:
        page_limit = st.selectbox("Rows per page", [25, 50, 100], index=1)

    total_evs, events_data = _load_audit_events_db(
        limit=page_limit,
        status=filter_status,
        search_name=search_name,
    )

    st.subheader(f"Audit Trail ({total_evs} matching events)")

    if events_data:
        df = pd.DataFrame(events_data)
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        
        display_cols = [c for c in ["id", "time", "camera_id", "status", "matched_name", "confidence", "distance", "liveness_score"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No audit records matching your criteria.")

    st.divider()
    st.subheader("Production Monitoring (Prometheus Metrics)")
    st.caption("Scrape URL: `http://localhost:8003/metrics` for Prometheus / Grafana integration.")
    if st.button("📊 View Live Prometheus Metrics"):
        resp = _api("/metrics")
        if resp.status_code == 200:
            st.code(resp.text, language="promql")
        else:
            st.error(f"Failed to fetch metrics: {resp.text}")
