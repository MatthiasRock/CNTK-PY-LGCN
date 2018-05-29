@echo off

:: Activate conda environment
call %ANACONDAPATH%\Scripts\activate %ANACONDAPATH%\envs\cntk-py35

python train.py
pause