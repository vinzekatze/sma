@echo off
:: Запускать из корня проекта (папки sma\)
:: Результаты появятся в research\docker_27\output\

setlocal

set IMAGE=sma-gmm27
set OUTPUT=%~dp0output

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

echo [1/2] Building image...
docker build -f research/docker_27/Dockerfile -t %IMAGE% .
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo [2/2] Running...
echo Results: %OUTPUT%
echo.

docker run --rm -v "%OUTPUT%:/output" -v "%~dp0..\..\data:/app/data:ro" %IMAGE%

echo.
echo Done. Check research\docker_27\output\27_report.txt
endlocal
