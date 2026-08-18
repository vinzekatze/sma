# Build context: project root (sma/)
#   docker build -f tests/Dockerfile -t sma-bench .
#
# Run (output to tests\output\ on host):
#   docker run --rm -v "%cd%\tests\output:/output" sma-bench

FROM python:3.11-slim

WORKDIR /app

# Slim deps — no streamlit/plotly needed
COPY tests/requirements_bench.txt /app/requirements_bench.txt
RUN pip install --no-cache-dir -r requirements_bench.txt

# Package
COPY sma/ /app/sma/

# Data
COPY data/candles/SBER/1d.json /app/data/candles/SBER/1d.json

# Benchmark script
COPY tests/benchmark_workers.py /app/benchmark_workers.py

VOLUME /output

CMD ["python", "-u", "benchmark_workers.py"]
