@echo off
REM Setup script for AgroDetect AI on Windows

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete!
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the Streamlit app:
echo   streamlit run src/ui/app.py
echo.
echo To run tests:
echo   pytest
