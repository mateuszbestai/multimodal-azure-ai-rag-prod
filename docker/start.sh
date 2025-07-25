#!/bin/bash

# docker/start.sh - Start both nginx and backend

echo "Starting backend API..."
cd /app
gunicorn --bind 127.0.0.1:8000 --workers 4 --threads 2 --timeout 300 --access-logfile - --error-logfile - wsgi:app &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend failed to start"
    exit 1
fi

echo "Starting nginx..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down..."
    kill $BACKEND_PID
    kill $NGINX_PID
    exit 0
}

# Trap signals
trap shutdown SIGTERM SIGINT

# Keep script running and monitor processes
while true; do
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "Backend process died, exiting..."
        kill $NGINX_PID
        exit 1
    fi
    
    # Check if nginx is still running
    if ! kill -0 $NGINX_PID 2>/dev/null; then
        echo "Nginx process died, exiting..."
        kill $BACKEND_PID
        exit 1
    fi
    
    sleep 10
done