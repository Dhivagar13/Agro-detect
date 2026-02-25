@echo off
echo ============================================================
echo AgroDetect AI - Quick Training Script
echo ============================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo ============================================================
echo Training Configuration
echo ============================================================
echo.
echo Please enter the path to your dataset folder:
echo Example: D:\datasets\plantvillage
echo Or: data\raw\plantvillage
echo.
set /p DATASET_PATH="Dataset path: "

echo.
echo How many disease classes are in your dataset?
echo (PlantVillage has 38 classes)
echo.
set /p NUM_CLASSES="Number of classes: "

echo.
echo How many epochs to train?
echo (Recommended: 50 for good accuracy, 20 for quick test)
echo.
set /p EPOCHS="Number of epochs: "

echo.
echo ============================================================
echo Starting Training...
echo ============================================================
echo.
echo Dataset: %DATASET_PATH%
echo Classes: %NUM_CLASSES%
echo Epochs: %EPOCHS%
echo.
echo This will take 1-5 hours depending on your hardware.
echo You can monitor progress in this window.
echo.
pause

python train_model.py --data-dir "%DATASET_PATH%" --num-classes %NUM_CLASSES% --epochs %EPOCHS% --batch-size 32

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo Your trained model is saved in the 'models' folder.
echo.
echo Next steps:
echo 1. Close this window
echo 2. Restart the Streamlit app
echo 3. Upload plant images to test your trained model!
echo.
echo To restart Streamlit, run: streamlit run src\ui\app.py
echo.
pause
