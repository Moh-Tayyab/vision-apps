# 📱 Mobile Camera Live Stream Connection Guide (USB Tethering)

A complete step-by-step guide to connect your mobile phone camera to the **Top Camera Carton Counter** system via USB cable for ultra-low latency, high frame-rate streaming.

---

## ⚡ Why USB Tethering?
- **Zero Latency / No Wi-Fi Lag:** Data transfers directly through high-speed USB wire instead of congested wireless networks.
- **Stable Bitrate:** Prevents frame drops, compression artifacts, and network disconnections during carton detection.
- **No Extra Drivers Required:** Native Android USB networking works out of the box on Linux/Windows/Mac.

---

## 📋 Prerequisites
1. **Android Smartphone** with a functional camera.
2. **USB Data Cable** (charging-only cables will not transfer video).
3. **IP Webcam App** installed on your phone (free on Google Play Store by *Pavel Khlebovich*).

---

## 🚀 Step-by-Step Instructions

### Step 1: Connect Wire & Enable USB Tethering
1. Plug your phone into your laptop/PC using the USB cable.
2. On your phone, open **Settings**.
3. Navigate to **Network & Internet** (or **Connections** / **Portable Hotspot** depending on your phone brand).
4. Tap **Hotspot & Tethering**.
5. Toggle **USB Tethering** to **ON**.

> 💡 *Note: If USB Tethering is greyed out, ensure the USB cable supports data transfer and is securely connected to both devices.*

---

### Step 2: Start IP Webcam Server on Mobile
1. Open the **IP Webcam** app on your phone.
2. *(Optional)* Under **Video preferences**, set resolution (e.g., `1280x720` or `1920x1080`) and quality (`80-90%`).
3. Scroll all the way to the bottom and tap **Start server**.
4. The camera view will open, and an IP address will appear at the bottom of the phone screen:
   ```
   http://192.168.42.129:8080
   ```
   *(Your IP digits may vary, e.g. `192.168.42.xxx:8080` or `192.168.43.xxx:8080`).*

---

### Step 3: Connect Camera on Web Dashboard
1. On your laptop, open your browser and go to:
   ```
   http://127.0.0.1:8001
   ```
2. In the **Camera Source Connect** section, enter the URL displayed on your phone, followed by `/video`:
   ```
   http://192.168.42.129:8080/video
   ```
3. Click the **Connect** button.
4. Once connected, click **Start Live Feed**.

You should now see the real-time overhead camera feed with bounding boxes, carton tracking, and layer status overlay!

---

## 🔄 Camera Mounting & Rotation Options

If your phone is mounted overhead in portrait or upside-down mode:
You can pass the rotation parameter via the `/camera/connect` API or IP Webcam app settings:
- **0°:** Standard landscape
- **90°:** Portrait mode
- **180°:** Inverted / upside-down mounting
- **270°:** Reverse portrait

---

## 🛠️ Alternative: Ultra-Fast ADB Port Forwarding

If you have **USB Debugging** enabled in Developer Options:
1. Connect via USB and run in your laptop terminal:
   ```bash
   adb forward tcp:8080 tcp:8080
   ```
2. Start IP Webcam on your phone.
3. In the dashboard, enter:
   ```
   http://127.0.0.1:8080/video
   ```
4. Click **Connect** (Direct localhost connection through USB wire).

---

## ❓ Troubleshooting & FAQs

### Q1: Dashboard says "Cannot open camera"
- Ensure **USB Tethering** is still enabled in phone settings (some phones turn it off if disconnected).
- Check that the phone screen says "Server is running" in the IP Webcam app.
- Make sure you appended `/video` at the end of the URL (e.g., `http://192.168.42.129:8080/video`).

### Q2: Stream has high latency or lag
- Lower the video resolution in the IP Webcam app to `1280x720` or `640x480`.
- Verify you are using USB Tethering (IP starts with `192.168.42.x`) rather than Wi-Fi.

### Q3: Phone battery drains quickly
- In the IP Webcam app, tap **Actions...** ➡️ **Lock in background** or turn down screen brightness while streaming.
