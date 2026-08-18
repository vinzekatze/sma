FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7"
COPY candles/ /app/data/candles/
COPY 59_schreiber_lwr.py /app/research/phase4_components/59_schreiber_lwr.py
RUN mkdir -p /app/research/figures
CMD ["sh", "-c", "python -u /app/research/phase4_components/59_schreiber_lwr.py 2>&1 | tee /app/research/figures/59_output.txt"]
