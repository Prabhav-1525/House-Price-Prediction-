@echo off
setlocal
REM Go to this script's folder
cd /d "%~dp0"

REM Create venv with Python 3.12 if it doesn't exist
if not exist .venv (
  py -3.12 -m venv .venv
)

REM Activate
call .venv\Scripts\activate

REM Upgrade tools
python -m pip install --upgrade pip setuptools wheel >nul

REM Install deps (only missing ones will be fetched)
pip install numpy==2.0.2 pandas==2.2.2 scikit-learn==1.4.2 matplotlib seaborn joblib streamlit

REM Run
streamlit run app1.py
