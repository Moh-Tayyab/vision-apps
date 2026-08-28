# Carton Counter — Complete Connection & Usage Guide

This guide provides step-by-step instructions for running the **Dual-Camera Layer-Wise Carton Counter** system in different production and testing environments:
1. **Wireless (Wi-Fi + Cloudflare Tunnel)** — *Recommended & Easiest (Zero SSL warnings)*
2. **Local Wi-Fi (Pure LAN / Offline)**
3. **Wired (USB Webcams / USB Cable Tethering)**
4. **REST API Direct Image Upload (`POST /count/dual`)**

---

## 🏗️ System Overview & Mathematics

The system utilizes two camera views:
* **Camera 1 (Front View):** Observes width and vertical layers ($N_{1,k}$).
* **Camera 2 (Side View at 90°):** Observes depth and vertical layers ($N_{2,k}$).

### Calculation Formula:
$$\text{Layer } k \text{ Count} = N_{1,k} \times N_{2,k}$$
$$\text{Total Pallet Count} = \sum_{k=1}^L (N_{1,k} \times N_{2,k})$$

---

## 🌐 Method 1: Wi-Fi + Cloudflare Tunnel *(Recommended)*

### Why Cloudflare Tunnel?
Mobile browsers (Google Chrome & Apple Safari) require an officially trusted HTTPS certificate to grant live camera access (`getUserMedia`). Cloudflare Tunnel generates an instant, valid, trusted public HTTPS link so your mobile cameras connect **without any certificate warnings**.

### Step-by-Step Instructions:
1. Open a terminal in the project directory and run:
   ```bash
   ./start_tunnel.sh
   ```
2. The script will automatically start the backend and print your active links:
   * **💻 Laptop Dashboard:** `https://<tunnel-id>.trycloudflare.com/`
   * **📱 Mobile 1 (Front Face):** `https://<tunnel-id>.trycloudflare.com/mobile?cam=cam1`
   * **📱 Mobile 2 (Side Face):** `https://<tunnel-id>.trycloudflare.com/mobile?cam=cam2`
   * **⚙️ Swagger API Docs:** `https://<tunnel-id>.trycloudflare.com/docs`
3. On **Mobile 1**, open the Mobile 1 link $\rightarrow$ Tap **"📹 Start Live Video Stream"** $\rightarrow$ Allow Camera.
4. On **Mobile 2**, open the Mobile 2 link $\rightarrow$ Tap **"📹 Start Live Video Stream"** $\rightarrow$ Allow Camera.
5. On your **Laptop**, open the Dashboard URL to view both live feeds, real-time bounding boxes, and the live layer breakdown table.

---

## 📶 Method 2: Direct Local Wi-Fi (No Internet / Pure LAN)

If your environment has no external internet connectivity:

1. Ensure both phones and the laptop are on the **same local Wi-Fi router** (or connect phones to your laptop's Mobile Hotspot).
2. Start the local server:
   ```bash
   PYTHONPATH=. ./.venv/bin/python apps/carton_counter/main.py
   ```
3. Find your laptop's local IP (e.g. `192.168.1.39` via `hostname -I`).
4. On **Mobile 1**, open:
   ```
   https://192.168.1.39:8443/mobile?cam=cam1
   ```
   * **Bypass Local SSL Warning:**
     * **Chrome (Android):** Tap **Advanced** $\rightarrow$ Tap **Proceed to 192.168.1.39 (unsafe)**.
     * **Safari (iOS):** Tap **Show Details** $\rightarrow$ Tap **visit this website**.
   * Tap **"📹 Start Live Video Stream"** $\rightarrow$ Tap **Allow**.
5. On **Mobile 2**, open:
   ```
   https://192.168.1.39:8443/mobile?cam=cam2
   ```
6. On **Laptop**, open:
   ```
   http://localhost:8001/
   ```

---

## 🔌 Method 3: Wired Connection (USB Webcams / USB Cables)

### Option A: External USB Webcams Plugged Directly into Laptop
1. Plug USB Webcam 1 (Front Face) and USB Webcam 2 (Side Face) into your laptop's USB ports.
2. Run the wired helper script:
   ```bash
   ./start_wired.sh
   ```
3. Open the Laptop Dashboard: **[http://localhost:8001/](http://localhost:8001/)**.
4. In the bottom panel, click the **🔌 Wired (USB Webcam)** tab.
5. Click **"🔍 Detect Cameras"**.
6. Assign and start devices:
   * **Camera 1 Device:** Select `/dev/video0` $\rightarrow$ Click **▶ Start**
   * **Camera 2 Device:** Select `/dev/video2` $\rightarrow$ Click **▶ Start**

---

### Option B: Mobile Phones via USB Cable (USB Tethering Mode)
To eliminate wireless Wi-Fi latency using mobile phones:
1. Connect Mobile 1 and Mobile 2 to the laptop using USB data cables.
2. On each phone, go to **Settings $\rightarrow$ Connections / Network $\rightarrow$ USB Tethering $\rightarrow$ Turn ON**.
3. Start the server:
   ```bash
   ./start_wired.sh
   ```
4. Open the mobile links using the tethered gateway IP.

---

## ⚙️ Mobile Browser Camera Permission Settings

If your mobile phone blocks camera access or prompts for permissions, follow these specific settings for Android and iPhone:

### A. Google Chrome (Android Phone):
1. **Open the stream link** in Google Chrome on your phone.
2. In the top URL address bar, tap the **🔒 (Lock icon)** or the **Tune (Sliders icon)** located next to the website URL.
3. Tap on **Permissions** (or **Site settings**).
4. Tap on **Camera** and select **"Allow"**.
5. **Refresh / Reload** the browser tab.
6. Tap the green **"📹 Start Live Video Stream"** button.

---

### B. Safari (iPhone / iPad / iOS):
1. **Open the stream link** in Safari on your iPhone.
2. In the top/bottom URL address bar, tap the **`aA`** icon on the left side of the address bar.
3. Tap on **Website Settings**.
4. Under **Camera**, change the setting to **"Allow"** (or **"Ask"**).
5. Tap **Done** in the top right and **Refresh** the page.
6. *If camera access remains blocked globally on iOS:*
   * Open iPhone **Settings** app $\rightarrow$ Scroll down and tap **Safari**.
   * Scroll to the **Settings for Websites** section $\rightarrow$ Tap **Camera**.
   * Change access to **"Allow"** or **"Ask"**.
   * Return to Safari and tap **"📹 Start Live Video Stream"**.

---

### 📸 Instant One-Tap Photo Mode (No Permissions Needed):
If your mobile browser security policy blocks video streaming:
* Tap the blue **"📸 Snap Photo & Send to AI"** button on the `/mobile` screen.
* This directly opens your phone's native built-in camera app without requiring any browser web permissions.
* Capturing the photo automatically sends it to the AI backend and computes the layer count.


---

## 📡 Method 4: REST API Direct Image Upload (`POST /count/dual`)

For automated systems, scripts, or post-processing:

### API Request (cURL):
```bash
curl -X POST "http://localhost:8001/count/dual" \
  -F "front=@path/to/front_view.jpg" \
  -F "side=@path/to/side_view.jpg" \
  -F "confidence=0.36" \
  -F "annotate=true"
```

### Response JSON:
```json
{
  "total_count": 48,
  "layers_count": 4,
  "layers": [
    {
      "layer_index": 1,
      "front_count": 3,
      "side_count": 4,
      "layer_total": 12,
      "y_range_front": [98.5, 182.0],
      "y_range_side": [102.0, 185.0]
    },
    {
      "layer_index": 2,
      "front_count": 3,
      "side_count": 4,
      "layer_total": 12,
      "y_range_front": [248.0, 332.0],
      "y_range_side": [250.0, 335.0]
    },
    {
      "layer_index": 3,
      "front_count": 3,
      "side_count": 4,
      "layer_total": 12,
      "y_range_front": [398.0, 482.0],
      "y_range_side": [400.0, 485.0]
    },
    {
      "layer_index": 4,
      "front_count": 3,
      "side_count": 4,
      "layer_total": 12,
      "y_range_front": [548.0, 632.0],
      "y_range_side": [550.0, 635.0]
    }
  ],
  "front_count_raw": 12,
  "side_count_raw": 16,
  "processing_time_ms": 45.2,
  "method": "dual_layer_multiplication",
  "front_annotated_base64": "data:image/jpeg;base64,...",
  "side_annotated_base64": "data:image/jpeg;base64,..."
}
```

---

## 📜 Helper Scripts Reference

| Script | Purpose | Execution Command |
| :--- | :--- | :--- |
| `start_tunnel.sh` | **Wi-Fi + Cloudflare Tunnel** (Auto-generates clean, trusted HTTPS links) | `./start_tunnel.sh` |
| `start_wired.sh` | **Wired USB Mode** (Detects attached V4L2 webcams and starts server) | `./start_wired.sh` |
| `start.sh` | **Docker Compose Mode** (Launches all 3 Vision Apps + Tunnels) | `./start.sh` |
