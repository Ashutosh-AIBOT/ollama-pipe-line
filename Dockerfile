FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl wget ffmpeg zstd && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install piper-tts
RUN mkdir -p /app/voices && \
    wget -q -O /app/voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O /app/voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY start.sh .
RUN chmod +x start.sh
EXPOSE 7860
CMD ["./start.sh"]
