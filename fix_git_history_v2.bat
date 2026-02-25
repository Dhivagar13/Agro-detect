@echo off
echo Fixing Git history to remove API keys from commits...
echo.

REM Reset to the commit before the problematic ones
echo Step 1: Resetting to commit before API keys were added...
git reset --soft 83ea12f

REM Stage all current changes
echo Step 2: Staging all current changes...
git add .

REM Create a new clean commit
echo Step 3: Creating new clean commit...
git commit -m "Fix Streamlit Cloud deployment and add dual AI integration (cleaned)"

REM Force push to overwrite history
echo Step 4: Force pushing to GitHub...
git push -f origin main

echo.
echo Done! Git history has been cleaned.
pause
