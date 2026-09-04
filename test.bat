@echo off
echo Testing llm_generator.py
.\venv\Scripts\python.exe content\llm_generator.py
echo.
echo Testing test_content_modules.py
.\venv\Scripts\python.exe content\test_content_modules.py
