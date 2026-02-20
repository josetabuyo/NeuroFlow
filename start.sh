#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"

lsof -ti:8501 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

echo ""
echo "  NeuroFlow"
echo "  ─────────────────────────────────"
echo "  Frontend → http://localhost:5173"
echo "  Backend  → http://localhost:8501"
echo "  ─────────────────────────────────"
echo ""

cd "$ROOT/backend" && source venv/bin/activate && uvicorn main:app --reload --port 8501 &

cd "$ROOT/frontend" && npm run dev &

wait
