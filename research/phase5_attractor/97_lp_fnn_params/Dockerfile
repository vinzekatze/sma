FROM python:3.11-slim

RUN pip install --no-cache-dir numpy scipy matplotlib

RUN mkdir -p /app/research/phase5_attractor/97_lp_fnn_params/results \
             /app/research/phase5_attractor/97_lp_fnn_params/figures \
             /app/data/candles

COPY 97_lp_fnn_params.py /app/research/phase5_attractor/97_lp_fnn_params/97_lp_fnn_params.py

CMD echo "=== 97_lp_fnn_params | v1.0 ===" && \
    echo "--- ТЕСТ (2 тикера, d=2..5, m_excess=0..5) ---" && \
    python /app/research/phase5_attractor/97_lp_fnn_params/97_lp_fnn_params.py --test && \
    echo "--- ПОЛНЫЙ ПРОГОН ---" && \
    python /app/research/phase5_attractor/97_lp_fnn_params/97_lp_fnn_params.py
