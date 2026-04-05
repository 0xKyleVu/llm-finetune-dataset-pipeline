FROM python:3.10-slim

WORKDIR /app

# Cài đặt Build dependencies và system libraries cho PDF parsing (docling)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Đẩy source code vào Container
COPY src/ ./src/

# Lúc này container chỉ nằm yên chờ Orchestrator hoặc lệnh gọi tay (idle state)
CMD ["tail", "-f", "/dev/null"]
