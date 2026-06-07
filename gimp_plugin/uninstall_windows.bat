@echo off
REM Spectrace GIMP Plugin Uninstaller (Windows)
REM Removes plugin and restores all original configs. No admin required.

setlocal enabledelayedexpansion

set "GIMP_DIR=%APPDATA%\GIMP\2.10"

echo === Spectrace GIMP Plugin Uninstaller (Windows) ===
echo.

REM Remove plugin
if exist "%GIMP_DIR%\plug-ins\spectrace_annotator.py" (
    del "%GIMP_DIR%\plug-ins\spectrace_annotator.py"
    echo [1/5] Removed plugin
) else (
    echo [1/5] Plugin not found (skipped)
)

REM Restore each config file from .original backup
set n=2
for %%f in (gimprc menurc toolrc sessionrc) do (
    if exist "%GIMP_DIR%\%%f.original" (
        move /y "%GIMP_DIR%\%%f.original" "%GIMP_DIR%\%%f" >nul
        echo [!n!/5] Restored original %%f
    ) else if exist "%GIMP_DIR%\%%f" (
        del "%GIMP_DIR%\%%f"
        echo [!n!/5] Removed spectrace %%f (GIMP will use defaults)
    ) else (
        echo [!n!/5] No %%f found (skipped)
    )
    set /a n+=1
)

echo.
echo === Done! Restart GIMP to complete uninstallation. ===

endlocal
