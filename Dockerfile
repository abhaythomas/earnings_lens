# EarningsLens v2 — Production Docker image
#
# Build:  docker build -t earningslens-v2 .
# Run:    docker run -p 8501:8501 \
#           -e GROQ_API_KEY=your_key \
#           -e PINECONE_API_KEY=your_key \
#           earningslens-v2
#
# Or with a .env file:
#         docker run -p 8501:8501 --env-file .env earningslens-v2

FROM python:3.11-slim

# Prevents Python from writing .pyc files and enables stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level deps needed by pdfplumber / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first (avoids downloading 2GB of CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (layer-cached unless requirements change)
COPY requirements_v2.txt .
RUN pip install --no-cache-dir -r requirements_v2.txt

# Copy only v2 source files — v1 files are NOT included
COPY app_v2.py .
COPY graph_v2.py .
COPY nodes_v2.py .
COPY ingest_v2.py .

# Copy data directory if it exists (transcripts / PDFs)
# If you mount a volume at runtime instead, remove or comment this out
COPY data/ ./data/

# Streamlit config — disable the browser auto-open and set server options
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n' \
    > /root/.streamlit/config.toml

EXPOSE 8501

# Health check — confirms Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
