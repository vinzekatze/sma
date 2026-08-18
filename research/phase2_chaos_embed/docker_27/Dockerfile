# Build context: project root (sma/)
#   docker build -f research/docker_27/Dockerfile -t sma-gmm27 .
#
# Run (results to research/docker_27/output/ on host):
#   docker run --rm -v "%cd%\research\docker_27\output:/output" sma-gmm27

FROM python:3.11-slim

WORKDIR /app

COPY research/docker_27/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Package
COPY sma/ /app/sma/

# Data монтируется снаружи при запуске (-v host_data:/app/data)
# Script
COPY research/docker_27/script.py /app/script.py

VOLUME /output

CMD ["python", "-u", "script.py"]
