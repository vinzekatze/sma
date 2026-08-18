FROM python:3.13-slim

WORKDIR /app

# Build deps (gcc needed for some numpy/pandas wheels on slim)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY sma/ ./sma/

# Data dir: mounted at runtime (DB + candles cache)
RUN mkdir -p /app/data

ENV SMA_DB_PATH=/app/data/sma.db

EXPOSE 8000

CMD ["uvicorn", "sma.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
