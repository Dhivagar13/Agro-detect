@echo off
echo ============================================================
echo  Git History Fix - Remove API Key from History
echo ============================================================
echo.

echo Step 1: Resetting to the commit before the API key was added...
git reset --soft 36aae5e

echo.
echo Step 2: Staging all current changes...
git add .

echo.
echo Step 3: Creating a new clean commit...
git commit -m "feat: Complete AgroDetect AI implementation with secure API key handling"

echo.
echo Step 4: Force pushing to overwrite history...
echo WARNING: This will rewrite Git history!
echo.
set /p confirm="Are you sure you want to continue? (yes/no): "

if /i "%confirm%"=="yes" (
    git push --force origin main
    echo.
    echo ============================================================
    echo  SUCCESS! Git history has been rewritten.
    echo  The API key has been removed from history.
    echo ============================================================
) else (
    echo.
    echo Operation cancelled. No changes made.
    echo.
    echo To undo the reset, run: git reset --hard origin/main
)

echo.
pause
