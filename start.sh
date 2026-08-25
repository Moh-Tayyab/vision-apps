#!/bin/bash
set -e

echo "============================================"
echo "   Vision Apps - Starting All Services"
echo "============================================"
echo ""

# Start docker compose
echo "[1/2] Starting Docker containers..."
docker compose up --build -d

echo ""
echo "[2/2] Starting Cloudflare Tunnels..."
echo ""

# Kill any existing cloudflared tunnels
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 2

# Start 3 separate tunnels
echo "--- Starting Carton Counter Tunnel (port 8001) ---"
nohup ./cloudflared tunnel --url http://localhost:8001 > /tmp/tunnel_carton.log 2>&1 &

echo "--- Starting Helmet Detection Tunnel (port 8002) ---"
nohup ./cloudflared tunnel --url http://localhost:8002 > /tmp/tunnel_helmet.log 2>&1 &

echo "--- Starting Face Authorization Tunnel (port 8003) ---"
nohup ./cloudflared tunnel --url http://localhost:8003 > /tmp/tunnel_face.log 2>&1 &

# Wait for tunnels to register
echo ""
echo "Waiting for tunnels to connect..."
sleep 8

# Extract URLs
CARTON_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/tunnel_carton.log | head -1)
HELMET_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/tunnel_helmet.log | head -1)
FACE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/tunnel_face.log | head -1)

echo ""
echo "============================================"
echo "   ALL SERVICES ARE LIVE!"
echo "============================================"
echo ""
echo "App 1 - Carton Counter:"
echo "  Dashboard : ${CARTON_URL}/"
echo "  Mobile    : ${CARTON_URL}/mobile"
echo "  API Docs  : ${CARTON_URL}/docs"
echo ""
echo "App 2 - Helmet Detection:"
echo "  Dashboard : ${HELMET_URL}/"
echo "  Mobile    : ${HELMET_URL}/mobile"
echo "  API Docs  : ${HELMET_URL}/docs"
echo ""
echo "App 3 - Face Authorization:"
echo "  Dashboard : ${FACE_URL}/"
echo "  API Docs  : ${FACE_URL}/docs"
echo ""
echo "============================================"
echo "  Press Ctrl+C to stop all services"
echo "============================================"

# Keep script running
trap "echo 'Stopping...'; docker compose down; pkill -f 'cloudflared tunnel'; exit 0" SIGINT SIGTERM

while true; do sleep 10; done
