#!/bin/bash
echo "🚀 Starting Cloudflare Live HTTPS Tunnel for Carton Counter (Port 8001)..."
if [ ! -f "./cloudflared" ]; then
    echo "📥 Downloading cloudflared binary..."
    curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared
fi
./cloudflared tunnel --url http://localhost:8001
