@echo off
:: Run from the PROJECT ROOT (sma\), not from tests\
:: Results will appear in tests\output\

setlocal

set IMAGE=sma-bench
set OUTPUT=%~dp0output

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

echo [1/2] Building image...
docker build -f tests/Dockerfile -t %IMAGE% .
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo [2/2] Running benchmark...
echo Results will be saved to: %OUTPUT%
echo.

docker run --rm -v "%OUTPUT%:/output" %IMAGE%

echo.
echo Done. Check tests\output\benchmark_results.json
endlocal
