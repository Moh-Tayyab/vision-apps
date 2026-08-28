#!/bin/bash
set -e

echo "============================================================"
echo "   🔌 Carton Counter — Wired USB / Local Mode"
echo "============================================================"
echo ""

# 1. Detect attached V4L2 USB camera devices
echo "🔍 Checking attached USB / Video devices:"
if ls /dev/video* 1> /dev/null 2>&1; then
    for dev in /dev/video*; do
        idx=$(echo "$dev" | sed 's/\/dev\/video//')
        name="Unknown"
        if [ -f "/sys/class/video4linux/video${idx}/name" ]; then
            name=$(cat "/sys/class/video4linux/video${idx}/name")
        fi
        echo "   ✅ $dev — $name"
    done
else
    echo "   ⚠️ No /dev/video* devices found."
fi

echo ""
# 2. Check local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
echo "🌐 Local IP: ${LOCAL_IP}"
echo ""

# 3. Start server
echo "▶ Starting Carton Counter Server on port 8001 (HTTP) & 8443 (HTTPS)..."
PYTHONPATH=. ./.venv/bin/python apps/carton_counter/main.py
