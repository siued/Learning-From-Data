@echo off

REM Check if a Python file is provided
IF "%~1"=="" (
    echo Usage: %0 ^<python_file^> [output_dir]
    exit /B 1
)

SET PYTHON_FILE=%~1
SET BASE_DIR=%~dp1
SET TIMESTAMP=%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
SET TIMESTAMP=%TIMESTAMP: =0%

REM Set output directory: use second argument if provided, otherwise create a timestamped directory
IF NOT "%~2"=="" (
    SET OUT_DIR=%BASE_DIR%%~2
) ELSE (
    SET OUT_DIR=%BASE_DIR%output_%TIMESTAMP%
)

REM Create the output directory
mkdir "%OUT_DIR%"

SET OUT_FILE=%OUT_DIR%\output.txt
SET PYTHON_FILE_COPY=%OUT_DIR%\%~n1_copy.py

REM Run the Python file and save output to a file and display it in the console
(
    python "%PYTHON_FILE%"
) > "%OUT_FILE%" 2>&1

REM Display the content of the output file in the console
type "%OUT_FILE%"

REM Copy the Python file to the output directory
copy "%PYTHON_FILE%" "%PYTHON_FILE_COPY%"
