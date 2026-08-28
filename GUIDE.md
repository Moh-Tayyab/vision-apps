# 📱 Multi-Camera & System Operation Guide

This comprehensive guide explains how to connect mobile phone cameras, USB webcams, and network streams to the **Vision Apps Enterprise Suite**, which contains 3 independent computer vision applications:

1. 📦 **App 1: Carton Counter** (Dual-Camera Layer-Wise Pallet Counting) — Port `8001` (HTTPS `8443`)
2. 🪖 **App 2: Helmet Safety Detection** (Real-Time PPE Compliance & Violation Logging) — Port `8002` (HTTPS `8444`)
3. 🔐 **App 3: Face Authorization & Access Control** (Biometric Gating + Streamlit Admin Portal) — Port `8003` (HTTPS `8445`), Streamlit `8501`

---

## 📑 Table of Contents
- [1. Quick Start: Launching Everything in 1 Command](#1-quick-start-launching-everything-in-1-command)
- [2. Universal Camera Connection Methods](#2-universal-camera-connection-methods)
  - [Method 1: Cloudflare Live HTTPS Tunnel (Recommended - Zero SSL Issues)](#method-1-cloudflare-live-https-tunnel-recommended---zero-ssl-issues)
  - [Method 2: Local Wi-Fi / Mobile Hotspot (Pure LAN / Offline)](#method-2-local-wi-fi--mobile-hotspot-pure-lan--offline)
  - [Method 3: Third-Party IP Camera Apps (IP Webcam / DroidCam / RTSP)](#method-3-third-party-ip-camera-apps-ip-webcam--droidcam--rtsp)
  - [Method 4: Wired USB Webcams & USB Tethering](#method-4-wired-usb-webcams--usb-tethering)
- [3. App 1: Carton Counter (Dual-Camera Pallet Counting)](#3-app-1-carton-counter-dual-camera-pallet-counting)
  - [Camera Placement & Mathematics](#camera-placement--mathematics)
  - [Connecting Mobile Cameras for App 1](#connecting-mobile-cameras-for-app-1)
  - [Using the Live Dual-View Dashboard](#using-the-live-dual-view-dashboard)
  - [Snapshot & REST API Upload](#snapshot--rest-api-upload)
- [4. App 2: Helmet Safety Detection (PPE Compliance)](#4-app-2-helmet-safety-detection-ppe-compliance)
  - [Connecting Mobile Camera for App 2](#connecting-mobile-camera-for-app-2)
  - [Safety Dashboard & Violation Logs](#safety-dashboard--violation-logs)
- [5. App 3: Face Authorization & Access Control](#5-app-3-face-authorization--access-control)
  - [Connecting Mobile Camera for App 3](#connecting-mobile-camera-for-app-3)
  - [Streamlit Admin Portal (Port 8501)](#streamlit-admin-portal-port-8501)
  - [FastAPI Live Access Gate (Port 8003)](#fastapi-live-access-gate-port-8003)
- [6. Mobile Browser Camera Permissions & Settings](#6-mobile-browser-camera-permissions--settings)
- [7. Troubleshooting & FAQ](#7-troubleshooting--faq)
- [8. Service Port & URL Reference Summary](#8-service-port--url-reference-summary)

---

## 1. Quick Start: Launching Everything in 1 Command

To launch all 3 applications with auto-generated Cloudflare HTTPS tunnels for your mobile phones:

```bash
./start.sh
```

This script will:
1. Build and start all 3 Docker containers (`carton-counter`, `helmet-detection`, `face-authorization`, and `face-auth-ui`).
2. Generate official, secure Cloudflare public HTTPS links.
3. Print ready-to-use Dashboard and Mobile links directly to your terminal.

```
============================================
   ALL SERVICES ARE LIVE!
============================================

App 1 - Carton Counter:
  💻 Dashboard : https://<tunnel-1>.trycloudflare.com/
  📱 Mobile 1  : https://<tunnel-1>.trycloudflare.com/mobile?cam=cam1
  📱 Mobile 2  : https://<tunnel-1>.trycloudflare.com/mobile?cam=cam2
  ⚙️ API Docs  : https://<tunnel-1>.trycloudflare.com/docs

App 2 - Helmet Detection:
  💻 Dashboard : https://<tunnel-2>.trycloudflare.com/
  📱 Mobile    : https://<tunnel-2>.trycloudflare.com/mobile
  ⚙️ API Docs  : https://<tunnel-2>.trycloudflare.com/docs

App 3 - Face Authorization:
  💻 Admin UI  : http://localhost:8501 (Streamlit)
  💻 Dashboard : https://<tunnel-3>.trycloudflare.com/
  📱 Mobile    : https://<tunnel-3>.trycloudflare.com/mobile
  ⚙️ API Docs  : https://<tunnel-3>.trycloudflare.com/docs
```

---

## 2. Universal Camera Connection Methods

Modern mobile browsers (Google Chrome on Android, Safari on iOS) **require HTTPS** to allow web applications to access the device camera (`navigator.mediaDevices.getUserMedia`). Below are the 4 ways you can connect your cameras:

```
+-----------------------------------------------------------------------------------+
|                            CAMERA CONNECTION MODES                                |
+-----------------------------------------------------------------------------------+
| [1] Cloudflare Tunnel (HTTPS)  -> No SSL warnings, 1-click camera access (Easiest)|
| [2] Local Wi-Fi / Hotspot HTTPS-> Self-signed SSL, bypass warning once in browser |
| [3] External IP Webcam App     -> Push MJPEG / RTSP stream URL to backend         |
| [4] Wired USB / Tethering      -> Plug physical USB webcam directly into PC/Server|
+-----------------------------------------------------------------------------------+
```

---

### Method 1: Cloudflare Live HTTPS Tunnel *(Recommended — Easiest)*

**Why use it?** Cloudflare creates an authenticated SSL certificate on the fly. Mobile Chrome and iOS Safari trust it instantly, giving seamless 1-tap camera access without certificate warnings.

#### For all 3 apps together:
```bash
./start.sh
```

#### For App 1 (Carton Counter only):
```bash
./start_tunnel.sh
```

#### How to use on mobile:
1. Open the printed `trycloudflare.com/mobile` link in your mobile browser.
2. Tap the green **"📹 Start Live Video Stream"** button.
3. Tap **"Allow"** on the browser camera permission prompt.
4. Open the Laptop Dashboard URL to view the live AI detections in real time.

---

### Method 2: Local Wi-Fi / Mobile Hotspot *(Pure LAN / Offline)*

If you are working in an offline factory or private network without internet access:

1. Connect your laptop and phone(s) to the **same Wi-Fi router** (or connect phones to your laptop's **Mobile Hotspot**).
2. Check your laptop's local IP address (e.g., `192.168.1.39` via `hostname -I` in Linux, or `ipconfig` in Windows).
3. Start the application backend locally:
   ```bash
   # App 1 (Carton Counter)
   PYTHONPATH=. ./.venv/bin/python apps/carton_counter/main.py
   
   # App 2 (Helmet Detection)
   PYTHONPATH=. ./.venv/bin/python apps/helmet_detection/main.py
   
   # App 3 (Face Authorization)
   PYTHONPATH=. ./.venv/bin/python apps/face_authorization/main.py
   ```
4. Open the dedicated HTTPS port in your phone's browser:
   * **App 1:** `https://192.168.1.39:8443/mobile`
   * **App 2:** `https://192.168.1.39:8444/mobile`
   * **App 3:** `https://192.168.1.39:8445/mobile`

#### How to Bypass the Local Self-Signed SSL Warning:
* **Android (Chrome):** Tap **Advanced** $\rightarrow$ Tap **"Proceed to 192.168.1.X (unsafe)"**.
* **iOS (Safari):** Tap **Show Details** $\rightarrow$ Tap **"visit this website"** $\rightarrow$ Confirm with FaceID / Passcode.
* Once loaded, tap **"📹 Start Live Video Stream"** and grant camera permission.

---

### Method 3: Third-Party IP Camera Apps (IP Webcam / DroidCam / RTSP)

You can also use dedicated mobile IP Camera apps installed on Android or iOS:

1. Install **IP Webcam** (Android) or **DroidCam Webcam** (Android / iOS) from the App Store / Play Store.
2. In the app, tap **"Start Server"**. The app will display an RTSP or HTTP MJPEG URL (e.g., `http://192.168.1.55:8080/video`).
3. Set this URL in your application environment or stream ingestion:
   ```bash
   export VIDEO_SOURCE="http://192.168.1.55:8080/video"
   ```
4. The backend will automatically ingest frames from the IP camera stream.

---

### Method 4: Wired USB Webcams & USB Tethering

#### Option A: External USB Webcams Plugged into Laptop
1. Plug USB Webcam 1 and USB Webcam 2 into your laptop's USB ports.
2. Run the wired startup script:
   ```bash
   ./start_wired.sh
   ```
3. Open the Laptop Dashboard: `http://localhost:8001/`.
4. Navigate to the **🔌 Wired (USB Webcam)** tab $\rightarrow$ Click **"🔍 Detect Cameras"**.
5. Select `/dev/video0` for Camera 1 and `/dev/video2` for Camera 2, then click **▶ Start**.

#### Option B: Mobile Phones via USB Cable (USB Tethering)
1. Connect your smartphone to the PC with a USB data cable.
2. On your phone: Go to **Settings $\rightarrow$ Network / Connections $\rightarrow$ USB Tethering $\rightarrow$ Turn ON**.
3. Use the tethered gateway IP displayed in your network settings to open `/mobile`.

---

## 3. App 1: Carton Counter (Dual-Camera Pallet Counting)

### Camera Placement & Mathematics

In mixed or multi-layer pallets, a single camera can miss stacked or occluded cartons. The Carton Counter solves this using **Dual-Angle Multi-View Layer Fusion**:

```
                 +-------------------+
                 |    TOP VIEW       |
                 +-------------------+
                           ▲
                           │
       +-------------------┼-------------------+
       |                   │                   |
   ┌───┴───┐               │               ┌───┴───┐
   │ Camera 1              │               │ Camera 2
   │ FRONT VIEW            │               │ SIDE VIEW (90°)
   │ (Detects width $N_1$) │               │ (Detects depth $N_2$)
   └───────┬───────────────┴───────────────┴───────┘
           │
           ▼
   +-----------------------------------------------+
   |             PALLET LAYER FORMULA              |
   |   Layer k Count = N_{1,k} x N_{2,k}           |
   |   Total Pallet Count = SUM(N_{1,k} x N_{2,k}) |
   +-----------------------------------------------+
```

---

### Connecting Mobile Cameras for App 1

For dual-camera layer counting, you need two mobile phones (or 1 phone moved between angles):

1. **Phone 1 (Front View Camera):**
   * Mount Phone 1 facing the **Front Face** of the pallet.
   * Open: `https://<tunnel-url>/mobile?cam=cam1`
   * Tap **"📹 Start Live Video Stream"** $\rightarrow$ Allow camera access.
2. **Phone 2 (Side View Camera):**
   * Mount Phone 2 at **90 degrees** facing the **Side Face** of the pallet.
   * Open: `https://<tunnel-url>/mobile?cam=cam2`
   * Tap **"📹 Start Live Video Stream"** $\rightarrow$ Allow camera access.

---

### Using the Live Dual-View Dashboard

Open `http://localhost:8001/` (or your Cloudflare Dashboard URL):

* **Left Monitor:** Live Front View stream with green bounding boxes and layer boundary lines.
* **Right Monitor:** Live Side View stream with green bounding boxes and layer boundary lines.
* **Top Metric Cards:** Total Pallet Carton Count, Total Physical Layers detected, and AI Inference latency (ms).
* **Layer Breakdown Table:** Real-time table displaying:
  * Layer Index ($k = 1, 2, \dots, L$)
  * Front Visible Count ($N_{1,k}$)
  * Side Visible Count ($N_{2,k}$)
  * Calculated Layer Total ($N_{1,k} \times N_{2,k}$)

---

### Snapshot & REST API Upload

If you do not want to stream continuous video, you can upload static photos:

```bash
curl -X POST "http://localhost:8001/count/dual" \
  -F "front=@/path/to/front_pallet.jpg" \
  -F "side=@/path/to/side_pallet.jpg" \
  -F "confidence=0.36" \
  -F "annotate=true"
```

**JSON Response:**
```json
{
  "total_count": 48,
  "layers_count": 4,
  "layers": [
    {"layer_index": 1, "front_count": 3, "side_count": 4, "layer_total": 12},
    {"layer_index": 2, "front_count": 3, "side_count": 4, "layer_total": 12},
    {"layer_index": 3, "front_count": 3, "side_count": 4, "layer_total": 12},
    {"layer_index": 4, "front_count": 3, "side_count": 4, "layer_total": 12}
  ],
  "processing_time_ms": 42.8,
  "method": "dual_layer_multiplication"
}
```

---

## 4. App 2: Helmet Safety Detection (PPE Compliance)

App 2 monitors construction and warehouse workers in real time, detecting persons and verifying whether each worker is wearing a safety helmet / hard hat.

```
       [ Live Mobile / CCTV Feed ]
                  │
                  ▼
       [ Person & Helmet AI Engine ]
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
 [ SAFE WORKER ]       [ SAFETY VIOLATION ]
 (Helmet Detected)     (No Helmet Detected)
  -> Green Box          -> Red Flashing Alert Box
  -> Status: OK         -> Logged to /violations
```

---

### Connecting Mobile Camera for App 2

1. Mount your phone on a tripod facing the entrance gate, inspection lane, or work area.
2. Open the mobile link in your phone's browser:
   * **Via Cloudflare:** `https://<helmet-tunnel-url>/mobile`
   * **Via Local Wi-Fi:** `https://192.168.1.39:8444/mobile`
3. Tap **"📹 Start Live Video Stream"** $\rightarrow$ Allow Camera.

---

### Safety Dashboard & Violation Logs

Open `http://localhost:8002/` on your monitor:
* **Live Video Monitor:** Real-time AI stream with:
  * 🟩 **Green Box:** Worker wearing helmet (`helmet`, confidence > 85%).
  * 🟥 **Red Box:** Worker without helmet (`NO HELMET VIOLATION`).
* **KPI Metrics:** Total Persons in Frame, Safe Workers Count, Active Safety Violations.
* **Audit Trail:** Timestamped violation log with snapshot captures.

---

## 5. App 3: Face Authorization & Access Control

App 3 provides biometric face authentication using deepface / Facenet 512D embeddings. It matches faces in real time against enrolled personnel and flags intruders or unauthorized individuals.

---

### Connecting Mobile Camera for App 3

1. Open the mobile link on your phone:
   * **Via Cloudflare:** `https://<face-tunnel-url>/mobile`
   * **Via Local Wi-Fi:** `https://192.168.1.39:8445/mobile`
2. **Tab 1: Face Enrollment:** Enter Employee Name $\rightarrow$ Take a clear front-facing photo $\rightarrow$ Tap **Enroll**.
3. **Tab 2: Live Verification:** Tap **"📹 Start Live Video Stream"** to stream live camera frames directly into the authentication engine.

---

### Streamlit Admin Portal (Port 8501)

Open `http://localhost:8501` in your desktop browser:

| Section | Capabilities |
| :--- | :--- |
| **📊 Dashboard** | Overall system statistics (total enrolled users, total embeddings stored, access events breakdown). |
| **➕ Enroll User** | Register new staff members with their name and clear frontal photo. Generates facial vector embeddings. |
| **👥 Manage Users** | View ID cards with photo thumbnails for all registered staff. One-click user deletion / revocation. |
| **📹 Live Detection** | Multi-source verification panel supporting In-browser webcam, Mobile IP stream, or image test uploads. |

---

### FastAPI Live Access Gate (Port 8003)

Open `http://localhost:8003/`:
* **Live Access Monitor:** Displays live incoming frames with bounding boxes:
  * 🟢 **Green Box:** Authorized Person (e.g., `Muhammad Tayyab — Distance: 0.12`).
  * 🔴 **Red Box:** Unauthorized Person (`UNAUTHORIZED / UNKNOWN INTRUDER`).
* **Real-time Event Stream:** Audio-visual feedback on access grants and denials.

---

## 6. Mobile Browser Camera Permissions & Settings

If your mobile browser blocks the camera or fails to display video, follow these platform-specific steps:

### A. Google Chrome (Android)
1. Open the stream URL (`.../mobile`) in Google Chrome.
2. Tap the **🔒 (Lock icon)** or **Tune (Sliders icon)** in the left of the URL bar.
3. Tap **Permissions** (or **Site Settings**).
4. Tap **Camera** $\rightarrow$ Change from *Block* to **Allow**.
5. **Reload the page** and tap **"📹 Start Live Video Stream"**.

---

### B. Apple Safari (iOS / iPhone / iPad)
1. Open the stream URL in Safari.
2. Tap the **`aA`** icon in the address bar.
3. Tap **Website Settings**.
4. Set **Camera** to **Allow**.
5. Tap **Done** and refresh the page.
6. *If camera is globally disabled:* Go to iPhone **Settings $\rightarrow$ Safari $\rightarrow$ Camera $\rightarrow$ Set to Allow**.

---

### 📸 Instant Native Photo Mode (Zero Permission Fallback)
If security policies prohibit live WebRTC/browser streaming on your phone:
* Tap the blue **"📸 Snap Photo & Send to AI"** button on the `/mobile` screen.
* This invokes your phone's native camera app directly without requiring browser permissions.
* The photo is immediately sent to the AI engine for instant processing.

---

## 7. Troubleshooting & FAQ

### Q1: The mobile page shows a black screen or "Connecting..." indefinitely.
* **Fix:** Ensure you opened the **HTTPS** URL (either via Cloudflare or via port `8443`/`8444`/`8445`). Mobile browsers completely block `navigator.mediaDevices.getUserMedia` over unencrypted `http://` on external IPs.

### Q2: How do I rotate or flip the camera image?
* In your environment or `.env` file, set:
  ```ini
  CAMERA_TRANSFORM=rotate_90_cw   # options: none, flip_h, flip_v, rotate_90_cw, rotate_90_ccw, rotate_180
  ```

### Q3: My phone and laptop cannot discover each other on local Wi-Fi.
* Check if your Wi-Fi router has **AP Isolation (Client Isolation)** enabled. If so, turn on your laptop's **Mobile Hotspot** and connect your phone directly to your laptop's hotspot network.

### Q4: The video stream has high latency or lag.
* The default streaming engine runs with adaptive frame throttling (10–15 FPS) to maintain smooth real-time AI inference. Ensure you have a strong Wi-Fi signal or use Cloudflare Tunnel for optimal routing.

---

## 8. Service Port & URL Reference Summary

| Application | HTTP Port | HTTPS Port (Local SSL) | Cloudflare URL Path | Streamlit UI | Swagger Docs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **App 1: Carton Counter** | `8001` | `8443` | `${CARTON_URL}/` | — | `/docs` |
| **App 1: Mobile (Cam 1 / Cam 2)** | `8001` | `8443` | `${CARTON_URL}/mobile?cam=cam1` | — | — |
| **App 2: Helmet Detection** | `8002` | `8444` | `${HELMET_URL}/` | — | `/docs` |
| **App 2: Mobile Stream** | `8002` | `8444` | `${HELMET_URL}/mobile` | — | — |
| **App 3: Face Authorization** | `8003` | `8445` | `${FACE_URL}/` | `8501` | `/docs` |
| **App 3: Mobile Stream / Enroll** | `8003` | `8445` | `${FACE_URL}/mobile` | — | — |
