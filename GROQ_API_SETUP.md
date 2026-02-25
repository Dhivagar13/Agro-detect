# 🔑 Groq API Key Setup Guide

## Quick Setup

### Option 1: Environment Variable (Recommended)

1. **Create a `.env` file** in the project root:
   ```bash
   # Copy the example file
   copy .env.example .env
   ```

2. **Edit `.env` file** and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

3. **Launch the app:**
   ```bash
   streamlit run src\ui\app.py
   ```

The API key will be loaded automatically from the `.env` file.

### Option 2: Settings Page

1. **Launch the app:**
   ```bash
   streamlit run src\ui\app.py
   ```

2. **Navigate to Settings:**
   - Click "⚙️ Settings" in the sidebar
   - Go to "🧠 Model" tab

3. **Enter your API key:**
   - Find "AI Enhancement" section
   - Enter your Groq API key
   - Click "💾 Save Model Settings"

4. **Restart the app** to apply changes

### Option 3: System Environment Variable

**Windows:**
```cmd
setx GROQ_API_KEY "your_actual_groq_api_key_here"
```

**Linux/Mac:**
```bash
export GROQ_API_KEY="your_actual_groq_api_key_here"
```

Then restart your terminal and launch the app.

## Getting Your Groq API Key

1. **Visit:** [groq.com](https://groq.com)
2. **Sign up** for a free account
3. **Navigate to** API Keys section
4. **Generate** a new API key
5. **Copy** the key (starts with `gsk_`)

## Security Notes

⚠️ **Important:**
- Never commit your `.env` file to Git
- Never share your API key publicly
- The `.env` file is already in `.gitignore`
- Use environment variables for production

✅ **Safe:**
- `.env` file (local only, gitignored)
- System environment variables
- Settings page (saved locally)

❌ **Unsafe:**
- Hardcoding in source code
- Committing to Git
- Sharing in public repositories

## Verification

To verify your API key is loaded:

1. Launch the app
2. Go to Settings → Model tab
3. Check for "✅ AI Analysis: Enabled"

If you see "⚠️ AI Analysis: Disabled", the API key is not loaded.

## Troubleshooting

### API Key Not Loading

**Check `.env` file:**
```bash
# Make sure .env exists
dir .env

# Check content (Windows)
type .env

# Check content (Linux/Mac)
cat .env
```

**Verify format:**
```env
GROQ_API_KEY=gsk_your_key_here
```

No quotes, no spaces around `=`

### Still Not Working

1. **Restart the app completely**
2. **Check Settings → Model tab**
3. **Manually enter API key in Settings**
4. **Save and restart**

## For Deployment

### Streamlit Cloud

Add to your Streamlit secrets:

1. Go to app settings
2. Add secret:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

### Docker

Pass as environment variable:

```bash
docker run -e GROQ_API_KEY=your_key_here your-image
```

### Heroku

```bash
heroku config:set GROQ_API_KEY=your_key_here
```

## Summary

**Recommended Setup:**
1. Create `.env` file
2. Add `GROQ_API_KEY=your_key_here`
3. Launch app
4. Verify in Settings

**The `.env` file is safe** - it's gitignored and stays local!

---

**Need Help?**
- Check GROQ_AI_INTEGRATION.md for full documentation
- See SETTINGS_IMPLEMENTATION.md for settings details
