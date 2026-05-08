@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "CONFIG=python\rl\configs\stage5_side3_cpu_i5_12500h.json"
set "PYTHON_EXE=%ROOT%.conda\trilibgo-rl\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [TriLibGo] Stage-5 Side-3 Conv RL training
echo [TriLibGo] Config: %CONFIG%

if "%~1"=="" (
    echo [TriLibGo] Starting fresh run
    "%PYTHON_EXE%" -m python.rl.train --config "%CONFIG%"
) else (
    echo [TriLibGo] Resuming from: %~1
    "%PYTHON_EXE%" -m python.rl.train --config "%CONFIG%" --resume "%~1"
)

endlocal
