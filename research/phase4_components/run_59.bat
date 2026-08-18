@echo off
setlocal

set IMAGE=sma-schr59
set BUILD=%~dp0build59
set OUTPUT=%~dp0output59

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

echo [1/2] Building...
docker build -t %IMAGE% "%BUILD%"
if errorlevel 1 (echo Build failed. & exit /b 1)

echo.
echo [2/2] Running  (15-30 min)
echo       Output: %OUTPUT%
echo.
docker run --rm -v "%OUTPUT%:/app/research/figures" %IMAGE%

echo.
echo Done. Results: %OUTPUT%
endlocal
