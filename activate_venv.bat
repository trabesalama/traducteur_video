@echo off
rem Script: activate_venv.bat
rem Find a .venv\Scripts\activate.bat in the current or parent directories and call it.

setlocal enabledelayedexpansion
set "CUR=%cd%"
:searchloop
if exist "%CUR%\\.venv\\Scripts\\activate.bat" (
    echo Found: %CUR%\\.venv\\Scripts\\activate.bat
    call "%CUR%\\.venv\\Scripts\\activate.bat"
    echo Activated .venv in %CUR%
    goto :eof
)
for %%I in ("%CUR%") do set "PARENT=%%~dpI"
rem Remove trailing backslash
if "%PARENT:~-1%"=="\\" set "PARENT=%PARENT:~0,-1%"
if "%PARENT%"=="%CUR%" goto notfound
set "CUR=%PARENT%"
goto searchloop
:notfound
echo No .venv\Scripts\activate.bat found in current or parent folders.
endlocal
