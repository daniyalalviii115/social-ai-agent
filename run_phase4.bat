@echo off
echo ============================================================
echo   PHASE 4 - Full Execution
echo ============================================================
echo.
echo [1/4] Install Check...
.\venv\Scripts\python.exe -c "import skfuzzy; from skfuzzy import control; print('Phase 4 dependencies satisfied')"
if errorlevel 1 goto :err

echo.
echo [2/4] Running fuzzy_engine.py...
.\venv\Scripts\python.exe decision\fuzzy_engine.py
if errorlevel 1 goto :err

echo.
echo [3/4] Running genetic_optimizer.py...
.\venv\Scripts\python.exe decision\genetic_optimizer.py
if errorlevel 1 goto :err

echo.
echo [4/4] Running test_decision_modules.py...
.\venv\Scripts\python.exe decision\test_decision_modules.py
if errorlevel 1 goto :err

echo.
echo ============================================================
echo   ALL PHASE 4 STEPS COMPLETE
echo ============================================================
goto :end

:err
echo.
echo ERROR: A step failed. Check output above.
exit /b 1

:end
