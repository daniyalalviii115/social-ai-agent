@echo off
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.headless true > streamlit.log 2>&1
