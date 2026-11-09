@echo off
cd /d "%~dp0"
echo ============================================================
echo Starting NephroScan AI(TM) Clinical Workstation...
echo ============================================================
call .venv\Scripts\activate.bat
streamlit run app2.py
pause
