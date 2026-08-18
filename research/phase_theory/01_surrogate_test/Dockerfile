FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy scipy matplotlib pandas

COPY . .

ENV DATA_DIR=/app/data/candles
ENV RESULTS_DIR=/output
ENV PYTHONUNBUFFERED=1

# Тестовый прогон (2 тикера, 5 origins, 5 суррогатов, 2 типа) — ~3 мин
# При успехе — полный прогон.
CMD echo "=== [T01] TEST RUN ===" && \
    TEST_MODE=1 WORKERS=${WORKERS:-4} \
        python research/phase_theory/01_surrogate_test/T01_collect.py && \
    echo "=== [T01] TEST PASSED — FULL RUN ===" && \
    WORKERS=${WORKERS:-4} \
        python research/phase_theory/01_surrogate_test/T01_collect.py && \
    echo "=== [T01] ANALYSIS ===" && \
    python research/phase_theory/01_surrogate_test/T02_analyze.py /output
