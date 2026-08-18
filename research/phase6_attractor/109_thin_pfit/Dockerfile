# Build context: project root (sma/)
#   docker build -f research/phase6_attractor/109_thin_pfit/Dockerfile \
#                -t thin-pfit-109 .
#
# Run (результаты пишутся на хост в results/, онлайн по мере готовности):
#   docker run --rm \
#       -v "$(pwd)/research/phase6_attractor/109_thin_pfit/results:/output" \
#       -e WORKERS=6 \
#       thin-pfit-109
#
# При старте контейнер печатает ID эксперимента и версию образа, затем
# автоматически запускает тестовый прогон (2 тикера, 5 origins, урезанная
# сетка — 1-2 минуты) и, если он не упал, полный прогон (8 тикеров,
# 65 комбинаций (n_iter,d), 40 origins, 20800 задач).

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy>=1.24.0 scipy>=1.11.0

COPY research/phase6_attractor/109_thin_pfit/109_thin_pfit.py /app/109_thin_pfit.py

# Только нужные тикеры/интервал — не весь data/candles
COPY data/candles/CHMF/1d.json /app/data/candles/CHMF/1d.json
COPY data/candles/LKOH/1d.json /app/data/candles/LKOH/1d.json
COPY data/candles/MGNT/1d.json /app/data/candles/MGNT/1d.json
COPY data/candles/MRKP/1d.json /app/data/candles/MRKP/1d.json
COPY data/candles/NLMK/1d.json /app/data/candles/NLMK/1d.json
COPY data/candles/NVTK/1d.json /app/data/candles/NVTK/1d.json
COPY data/candles/SBER/1d.json /app/data/candles/SBER/1d.json
COPY data/candles/VTBR/1d.json /app/data/candles/VTBR/1d.json

ENV DATA_DIR=/app/data/candles
ENV RESULTS_DIR=/output
VOLUME /output

CMD ["sh", "-c", "\
    echo '=== 109_thin_pfit  image=v1 ===' && \
    echo '--- ТЕСТОВЫЙ ПРОГОН (2 тикера, 5 origins, урезанная сетка) ---' && \
    TEST_MODE=1 python -u /app/109_thin_pfit.py && \
    echo '--- ТЕСТ ПРОЙДЕН. РАБОЧИЙ ПРОГОН (8 тикеров, 65 комбинаций, 40 origins) ---' && \
    python -u /app/109_thin_pfit.py \
"]
