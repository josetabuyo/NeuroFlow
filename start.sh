#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load nvm if available so the correct Node version is used
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Default ports (different from original to avoid conflicts)
DEFAULT_FRONTEND_PORT=5180
DEFAULT_BACKEND_PORT=8510

# Parse command line arguments
FRONTEND_PORT=$DEFAULT_FRONTEND_PORT
BACKEND_PORT=$DEFAULT_BACKEND_PORT

while [[ $# -gt 0 ]]; do
  case $1 in
    --frontend-port)
      FRONTEND_PORT="$2"
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Start NeuroFlow backend and frontend"
      echo ""
      echo "Options:"
      echo "  --frontend-port PORT  Frontend port (default: $DEFAULT_FRONTEND_PORT)"
      echo "  --backend-port PORT   Backend port (default: $DEFAULT_BACKEND_PORT)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Kill any processes using the ports
lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null

echo ""
echo "  NeuroFlow"
echo "  ─────────────────────────────────"
echo "  Frontend → http://localhost:$FRONTEND_PORT"
echo "  Backend  → http://localhost:$BACKEND_PORT"
echo "  Docs     → http://localhost:$BACKEND_PORT/docs"
echo "  ─────────────────────────────────"
echo ""

# Update frontend configuration to use the correct backend port
sed -i.bak "s|target: 'http://localhost:[0-9]\+'|target: 'http://localhost:$BACKEND_PORT'|g" "$ROOT/frontend/vite.config.ts"
sed -i.bak "s|port: [0-9]\+|port: $FRONTEND_PORT|g" "$ROOT/frontend/vite.config.ts"

# Start backend
cd "$ROOT/backend" && source venv/bin/activate && uvicorn main:app --reload --port $BACKEND_PORT &

# Start frontend
cd "$ROOT/frontend" && npm run dev &

# Wait for both processes
wait

# Restore original configuration on exit
trap 'cd "$ROOT" && rm -f frontend/vite.config.ts.bak 2>/dev/null' EXIT
