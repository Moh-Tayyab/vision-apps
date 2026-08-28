#!/bin/bash
set -e

echo "============================================================"
echo "   🚀 Carton Counter — Cloudflare Live HTTPS Tunnel"
echo "============================================================"
echo ""

# 1. Start Carton Counter server if not already running
if ! curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "▶ Starting local Carton Counter backend on port 8001..."
    PYTHONPATH=. ./.venv/bin/python apps/carton_counter/main.py > /tmp/carton_counter.log 2>&1 &
    sleep 3
fi

# 2. Check cloudflared binary
if [ ! -f "./cloudflared" ]; then
    echo "📥 Downloading cloudflared binary..."
    curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared
fi

# 3. Kill old tunnels
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

# 4. Start tunnel in background and capture URL
echo "🌐 Creating Cloudflare HTTPS Tunnel..."
./cloudflared tunnel --url http://localhost:8001 > /tmp/cf_tunnel.log 2>&1 &

echo "⏳ Waiting for Cloudflare URL..."
TUNNEL_URL=""
for i in {1..15}; do
    sleep 1
    TUNNEL_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1 || true)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
done

if [ -z "$TUNNEL_URL" ]; then
    echo "⚠️ Tunnel starting, check log: cat /tmp/cf_tunnel.log"
    exit 1
fi

echo ""
echo "============================================================"
echo "   ✅ CLOUDFLARE LIVE LINKS (No SSL Warnings, 100% Trusted)"
echo "============================================================"
echo ""
echo "💻 Laptop Dashboard :"
echo "   👉 ${TUNNEL_URL}/"
echo ""
echo "📱 Mobile 1 (Camera 1 — Front Face):"
echo "   👉 ${TUNNEL_URL}/mobile?cam=cam1"
echo ""
echo "📱 Mobile 2 (Camera 2 — Side Face):"
echo "   👉 ${TUNNEL_URL}/mobile?cam=cam2"
echo ""
echo "⚙️ Swagger API Docs :"
echo "   👉 ${TUNNEL_URL}/docs"
echo ""
echo "============================================================"
echo "  Press Ctrl+C to stop the tunnel"
echo "============================================================"

trap "echo 'Stopping Cloudflare tunnel...'; pkill -f 'cloudflared tunnel'; exit 0" SIGINT SIGTERM

while true; do sleep 10; done
