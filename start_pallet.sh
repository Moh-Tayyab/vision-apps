#!/bin/bash
set -e

echo "============================================================"
echo "   📦 Pallet Counter — Row-Wise Carton Counter"
echo "============================================================"
echo ""

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
PORT=${PORT:-8004}

echo "🌐 Local IP: ${LOCAL_IP}"
echo "▶ Starting Pallet Counter Server on port ${PORT}..."
echo "📊 Dashboard : http://localhost:${PORT}/ or http://${LOCAL_IP}:${PORT}/"
echo "📖 API Docs  : http://localhost:${PORT}/docs"
echo ""

PYTHONPATH=apps/pallet_counter PORT=${PORT} ./.venv/bin/python apps/pallet_counter/main.py
