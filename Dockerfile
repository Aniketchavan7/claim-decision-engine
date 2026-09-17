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

# Set up user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Copy and install python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --default-timeout=1000 --retries 10 -r requirements.txt

# Pre-download local embedding & reranking models to avoid cold start download lag
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy repository content with proper ownership
COPY --chown=user . $HOME/app

# Ensure start.sh has executable permissions
RUN chmod +x $HOME/app/start.sh

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

CMD ["/home/user/app/start.sh"]

