# Production-Grade Multi-Stage Dockerfile for RAG NLP Intelligence Platform
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for C++ compilation and FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command launches FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
