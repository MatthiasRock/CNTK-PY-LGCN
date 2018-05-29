::########################################################################################
:: Execute this batch file if you want to use a CNTK development setup (after each build)
::########################################################################################
@echo off

:: Adapt these paths
::###########################################################
set CNTK_PATH=C:\repos\cntk
set ANACONDA_PATH=C:\local\Anaconda3-4.1.1-Windows-x86_64
::###########################################################

setx PYTHONPATH %CNTK_PATH%\bindings\python > nul 2> nul
setx CNTK_PY35_PATH %ANACONDA_PATH%\envs\cntk-py35 > nul 2> nul
setx ANACONDAPATH %ANACONDA_PATH% > nul 2> nul

:: Create & update conda environment
call %ANACONDA_PATH%\scripts\conda env create --file %CNTK_PATH%\Scripts\install\windows\conda-windows-cntk-py35-environment.yml  --name cntk-py35 > nul 2> nul
call %ANACONDA_PATH%\scripts\conda env update --file %CNTK_PATH%\Scripts\install\windows\conda-windows-cntk-py35-environment.yml  --name cntk-py35
pause