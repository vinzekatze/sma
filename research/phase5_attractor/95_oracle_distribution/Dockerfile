FROM python:3.11-slim

LABEL experiment="95a_oracle_distribution" version="1.0"

ENV EXPERIMENT_ID="95a_oracle_distribution"
ENV DOCKER_VERSION="1.0"

# Пути монтируются снаружи через -v
ENV DATA_DIR="/data"
ENV RESULTS_DIR="/results"
ENV FIGURES_DIR="/figures"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 95a_oracle_distribution.py .

# Создаём точки монтирования
RUN mkdir -p /data /results /figures

CMD echo "════════════════════════════════════════════════════" && \
    echo "  EXPERIMENT : ${EXPERIMENT_ID}" && \
    echo "  DOCKER VER : ${DOCKER_VERSION}" && \
    echo "════════════════════════════════════════════════════" && \
    echo "" && \
    echo ">>> ТЕСТОВЫЙ ПРОГОН (малые данные) ..." && \
    python 95a_oracle_distribution.py --mode test && \
    echo "" && \
    echo ">>> ТЕСТ ПРОЙДЕН. ЗАПУСК ПОЛНОГО ПРОГОНА ..." && \
    echo "" && \
    python 95a_oracle_distribution.py --mode full
