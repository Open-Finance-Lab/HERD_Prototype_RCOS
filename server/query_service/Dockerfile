FROM vllm/vllm-openai:latest

# Optional envs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_CUDA=1 \
    TOPIC_VECTOR_PATH=topic_vector.json \
    EMBED_MODEL_NAME=all-MiniLM-L6-v2

WORKDIR /app

# Install any additional Python dependencies (FastAPI, sentence-transformers, etc.)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code into the image
COPY . .

# Expose your router + vllm ports
EXPOSE 8000 8001

# Run both vLLM and your API server
CMD python3 -m vllm.entrypoints.openai.api_server \
      --model-id distilbert-base-uncased \
      --port 8001 & \
    uvicorn app:app --host 0.0.0.0 --port 8000
