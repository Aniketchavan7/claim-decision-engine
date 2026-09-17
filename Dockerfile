FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download local embedding & reranking models to avoid cold start download lag
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy repository content
COPY . .

# Expose port for Hugging Face Spaces / Container
EXPOSE 7860

# Create startup script to run FastAPI in background and Streamlit on port 7860
RUN echo '#!/bin/bash\n\
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &\n\
API_BASE_URL="http://localhost:8000" streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0\n'\
> /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
