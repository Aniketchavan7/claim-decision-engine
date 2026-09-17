#!/bin/bash
set -e

# Start FastAPI backend in background on port 8000
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for FastAPI backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "FastAPI backend is ready."
        break
    fi
    sleep 2
done

export API_BASE_URL="http://localhost:8000"
exec streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
