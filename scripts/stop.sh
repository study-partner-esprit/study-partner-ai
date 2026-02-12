#!/bin/bash

# Stop all Study Partner AI services

echo "🛑 Stopping Study Partner AI services..."

# Stop AI Backend
if [ -f .ai-backend.pid ]; then
    PID=$(cat .ai-backend.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping AI Backend (PID: $PID)..."
        kill $PID
        rm .ai-backend.pid
        echo "✓ AI Backend stopped"
    else
        echo "AI Backend process not found"
        rm .ai-backend.pid
    fi
else
    echo "No AI Backend PID file found"
    # Try to find and kill the process anyway
    pkill -f "services.api.main" && echo "✓ Killed AI Backend process"
fi

echo ""
echo "All services stopped"
