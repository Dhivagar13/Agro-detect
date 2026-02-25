# 🚀 Quick Fix Guide - Remove API Key from Git History

## Problem
The API key is in your Git commit history (commit `eab93e1`), so GitHub is blocking the push.

## Solution Options

### Option 1: Allow Secret on GitHub (Fastest - 2 minutes)

This is the quickest solution if you want to keep your commit history.

1. **Click the GitHub link** from the error message:
   ```
   https://github.com/Dhivagar13/Agro-detect/security/secret-scanning/unblock-secret/3A9GkWvQF69id4vJsO7VGEfQaUU
   ```

2. **Click "Allow secret"** button

3. **Push again:**
   ```bash
   git push origin main
   ```

**Pros:** Fast, keeps history  
**Cons:** API key remains in history (but marked as allowed)

### Option 2: Rewrite Git History (Recommended - 5 minutes)

This completely removes the API key from history.

#### Automatic Script:

```bash
./fix_git_history.bat
```

Then type `yes` when prompted.

#### Manual Steps:

```bash
# 1. Reset to before the API key commit
git reset --soft 36aae5e

# 2. Stage all changes
git add .

# 3. Create new clean commit
git commit -m "feat: Complete AgroDetect AI implementation with secure API key handling"

# 4. Force push (rewrites history)
git push --force origin main
```

**Pros:** Completely removes API key from history  
**Cons:** Rewrites history (force push required)

### Option 3: Revoke and Create New API Key (Most Secure - 10 minutes)

1. **Revoke the exposed API key:**
   - Go to [groq.com](https://groq.com)
   - Delete the old API key
   - Generate a new one

2. **Update your `.env` file:**
   ```env
   GROQ_API_KEY=your_new_api_key_here
   ```

3. **Use Option 2** to rewrite history

4. **Push:**
   ```bash
   git push --force origin main
   ```

**Pros:** Most secure, old key is invalid  
**Cons:** Takes longer, need new API key

## Recommended Approach

**For Quick Fix:** Use Option 1 (Allow secret on GitHub)

**For Security:** Use Option 3 (Revoke and create new key)

## Step-by-Step: Option 2 (Rewrite History)

### Windows:

1. **Open PowerShell in project directory**

2. **Run the fix script:**
   ```bash
   ./fix_git_history.bat
   ```

3. **Type `yes` when prompted**

4. **Done!** The API key is removed from history.

### Verification:

```bash
# Check that push succeeded
git log --oneline -3

# Should show new commit without API key history
```

## What This Does

1. **Resets** to commit `36aae5e` (before API key was added)
2. **Keeps all your changes** in working directory
3. **Creates new commit** without API key in history
4. **Force pushes** to overwrite remote history

## Important Notes

⚠️ **Force Push Warning:**
- This rewrites Git history
- If others have cloned the repo, they'll need to re-clone
- Only do this if you're the only one working on the repo

✅ **Safe Because:**
- Your `.env` file is gitignored
- Current code doesn't have hardcoded key
- Only removes from history

## After Fixing

1. **Verify push succeeded:**
   ```bash
   git status
   ```

2. **Check GitHub:**
   - No secret scanning alerts
   - Latest commit visible

3. **Test app:**
   ```bash
   streamlit run src\ui\app.py
   ```

4. **Verify AI works:**
   - Go to Settings → Model
   - Should show "✅ AI Analysis: Enabled"

## If Something Goes Wrong

**Undo the reset:**
```bash
git reset --hard origin/main
```

**Start over:**
```bash
git fetch origin
git reset --hard origin/main
```

## Need Help?

If you're unsure, use **Option 1** (Allow secret on GitHub) - it's the safest and fastest.

---

**Choose your option and let's get this fixed!** 🚀
