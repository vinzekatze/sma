#!/bin/bash
cd "$(dirname "$0")"
source /home/kali/.venvs/sma/bin/activate
exec uvicorn sma.api.app:app --reload --host 0.0.0.0 --port 8000
