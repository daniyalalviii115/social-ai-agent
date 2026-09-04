@echo off
echo ============================================================
echo   PHASE 5 - Full Execution
echo ============================================================
echo.
echo [1/3] Running llm_generator.py...
.\venv\Scripts\python.exe content\llm_generator.py
if errorlevel 1 goto :err

echo.
echo [2/3] Running banner_renderer.py...
.\venv\Scripts\python.exe content\banner_renderer.py
if errorlevel 1 goto :err

echo.
echo [3/3] Running test_content_modules.py...
.\venv\Scripts\python.exe content\test_content_modules.py
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   ALL PHASE 5 STEPS COMPLETE
echo ============================================================
goto :end

:err
echo.
echo ERROR: A step failed. Check output above.
exit /b 1

:end
