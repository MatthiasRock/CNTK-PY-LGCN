@echo off

:: Activate conda environment
call %ANACONDAPATH%\Scripts\activate %ANACONDAPATH%\envs\cntk-py35

python predict.py
pause