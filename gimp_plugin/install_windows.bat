@echo off
REM Spectrace GIMP Plugin Installer (Windows)
REM Installs the plugin, configures paths, and locks down the GIMP UI for annotation.
REM No admin required.

setlocal enabledelayedexpansion

set "GIMP_DIR=%APPDATA%\GIMP\2.10"

if not exist "%GIMP_DIR%" (
    echo ERROR: GIMP 2.10 config directory not found at:
    echo   %GIMP_DIR%
    echo Make sure GIMP 2.10 has been launched at least once.
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Resolve spectrace root (parent of gimp_plugin)
for %%I in ("%SCRIPT_DIR%\..") do set "SPECTRACE_ROOT=%%~fI"

echo === Spectrace GIMP Plugin Installer (Windows) ===
echo.
echo GIMP config:    %GIMP_DIR%
echo Spectrace root: %SPECTRACE_ROOT%
echo.

REM Back up config files (only on first install)
for %%f in (gimprc menurc toolrc sessionrc) do (
    if exist "%GIMP_DIR%\%%f" (
        if not exist "%GIMP_DIR%\%%f.original" (
            copy "%GIMP_DIR%\%%f" "%GIMP_DIR%\%%f.original" >nul
            echo   Backed up %%f -^> %%f.original
        )
    )
)
echo.

REM 1. Install plugin
echo [1/6] Installing plugin...
if not exist "%GIMP_DIR%\plug-ins" mkdir "%GIMP_DIR%\plug-ins"
copy /y "%SCRIPT_DIR%\spectrace_annotator.py" "%GIMP_DIR%\plug-ins\" >nul
echo   -^> Done

REM 2. Install gimprc
echo [2/6] Installing gimprc (hides menubar, rulers)...
if exist "%SCRIPT_DIR%\config\gimprc" (
    copy /y "%SCRIPT_DIR%\config\gimprc" "%GIMP_DIR%\" >nul
    echo   -^> Done
) else (
    echo   -^> Skipped (no gimprc found)
)

REM 3. Install stripped shortcuts
echo [3/6] Installing menurc (strips keyboard shortcuts)...
if exist "%SCRIPT_DIR%\config\menurc" (
    copy /y "%SCRIPT_DIR%\config\menurc" "%GIMP_DIR%\" >nul
    echo   -^> Done
) else (
    echo   -^> Skipped (no menurc found)
)

REM 4. Install stripped toolbox
echo [4/6] Installing toolrc (pencil + eraser only)...
if exist "%SCRIPT_DIR%\config\toolrc" (
    copy /y "%SCRIPT_DIR%\config\toolrc" "%GIMP_DIR%\" >nul
    echo   -^> Done
) else (
    echo   -^> Skipped (no toolrc found)
)

REM 5. Install minimal session layout
echo [5/6] Installing sessionrc (minimal dock layout)...
if exist "%SCRIPT_DIR%\config\sessionrc" (
    copy /y "%SCRIPT_DIR%\config\sessionrc" "%GIMP_DIR%\" >nul
    echo   -^> Done
) else (
    echo   -^> Skipped (no sessionrc found)
)

REM 6. Create spectrace configuration
echo [6/6] Creating spectrace configuration...
set "SPECTRACE_CONFIG_DIR=%USERPROFILE%\.spectrace"
set "SPECTRACE_CONFIG=%SPECTRACE_CONFIG_DIR%\config.json"
if not exist "%SPECTRACE_CONFIG_DIR%" mkdir "%SPECTRACE_CONFIG_DIR%"

REM Auto-detect conda Python 3
set "PYTHON3_PATH="
for %%B in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "C:\miniconda3" "C:\anaconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3") do (
    if exist "%%~B\envs\spectrace\python.exe" (
        set "PYTHON3_PATH=%%~B\envs\spectrace\python.exe"
        goto :found_python
    )
)

REM Try conda run
where conda >nul 2>nul
if !errorlevel! equ 0 (
    for /f "tokens=*" %%P in ('conda run -n spectrace where python 2^>nul') do (
        set "PYTHON3_PATH=%%P"
        goto :found_python
    )
)

:found_python
if "!PYTHON3_PATH!"=="" (
    echo   WARNING: Could not find spectrace conda environment.
    echo   WAV file opening will not work until you set python3_path in:
    echo     %SPECTRACE_CONFIG%
    echo.
    echo   To fix: conda activate spectrace ^& where python
    echo   Then edit the config file with that path.
    set "PYTHON3_PATH=python"
)

REM Escape backslashes for JSON
set "JSON_ROOT=!SPECTRACE_ROOT:\=\\!"
set "JSON_PYTHON=!PYTHON3_PATH:\=\\!"

REM Write config
(
echo {
echo   "spectrace_root": "!JSON_ROOT!",
echo   "python3_path": "!JSON_PYTHON!",
echo   "default_nfft": 2048,
echo   "default_grayscale": true,
echo   "default_project_dir": "projects"
echo }
) > "%SPECTRACE_CONFIG%"

echo   -^> Config written to: %SPECTRACE_CONFIG%
echo      spectrace_root: %SPECTRACE_ROOT%
echo      python3_path:   !PYTHON3_PATH!

echo.
echo === Installed! Close GIMP completely and reopen. ===
echo.
echo After restart you will see:
echo   - No menubar (right-click canvas for menus)
echo   - Only Pencil and Eraser in the toolbox
echo   - Only Tool Options (left) and Layers (right)
echo   - No brushes, patterns, fonts, or channels docks
echo.
echo New features:
echo   - File ^> Open ^> select a .wav file -^> opens as spectrogram
echo   - Filters ^> Spectrace ^> Setup Annotation -^> pick a template .xcf
echo.
echo Usage: right-click canvas ^> Filters ^> Spectrace ^> Setup Annotation...
echo.
echo To restore original GIMP: %SCRIPT_DIR%\uninstall_windows.bat

endlocal
