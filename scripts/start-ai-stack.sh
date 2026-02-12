#!/bin/bash

# Study Partner AI - Complete Stack Startup Script
# This script starts all services needed for testing the AI features

set -e

echo "🚀 Starting Study Partner AI Stack..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if MongoDB is running
echo -e "${YELLOW}Checking MongoDB...${NC}"
if ! pgrep -x "mongod" > /dev/null; then
    echo -e "${RED}❌ MongoDB is not running${NC}"
    echo "Please start MongoDB first:"
    echo "  sudo systemctl start mongod"
    echo "  OR"
    echo "  docker run -d -p 27017:27017 --name mongodb mongo:latest"
    exit 1
fi
echo -e "${GREEN}✓ MongoDB is running${NC}"
echo ""

# Start AI Backend (FastAPI)
echo -e "${YELLOW}Starting AI Backend (FastAPI on port 8000)...${NC}"

# Go to project root (parent of scripts directory)
cd "$(dirname "$0")/.."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start in background with logging
poetry run python -m services.api.main > logs/ai-backend.log 2>&1 &
AI_PID=$!
echo $AI_PID > .ai-backend.pid
echo -e "${GREEN}✓ AI Backend started (PID: $AI_PID)${NC}"
echo "  Logs: logs/ai-backend.log"
echo "  API Docs: http://localhost:8000/docs"
echo ""

# Wait for backend to start (with retries)
echo -e "${YELLOW}Waiting for AI Backend to be ready...${NC}"
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ AI Backend is responsive${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}❌ AI Backend failed to start after ${MAX_RETRIES} attempts${NC}"
        echo "Check logs at: logs/ai-backend.log"
        echo ""
        echo "Last 20 lines of log:"
        tail -20 logs/ai-backend.log
        exit 1
    fi
    sleep 2
    echo -n "."
done
echo ""

echo -e "${GREEN}=== All Services Started ===${NC}"
echo ""
echo "AI Backend:  http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo ""
echo "Next steps:"
echo "1. Start the React frontend:"
echo "   cd ../study-partner-web && npm run dev"
echo ""
echo "2. Visit http://localhost:5173 in your browser"
echo ""
echo "To stop services:"
echo "  ./scripts/stop.sh"
echo "  OR"
echo "  kill $(cat .ai-backend.pid)"
echo ""

# Keep script running to show logs
echo -e "${YELLOW}Showing AI Backend logs (Ctrl+C to exit):${NC}"
echo "---"
tail -f logs/ai-backend.log
