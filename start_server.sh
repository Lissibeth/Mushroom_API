#!/bin/bash
echo "Starting Mushroom Classification API Server..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload