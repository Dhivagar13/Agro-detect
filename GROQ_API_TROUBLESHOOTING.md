# 🔧 Groq API Troubleshooting Guide

## Common Errors and Solutions

### Error: "400 Client Error: Bad Request"

**Cause:** Invalid API key format or expired key

**Solutions:**

1. **Check API Key Format**
   - API key should start with `gsk_`
   - No spaces or quotes
   - Example: `gsk_abc123def456...`

2. **Verify API Key in `.env` file:**
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
   - No quotes around the key
   - No spaces before or after `=`

3. **Get a New API Key:**
   - Visit [groq.com](https://groq.com)
   - Sign in to your account
   - Go to API Keys section
   - Delete old key
   - Generate new key
   - Copy the new key

4. **Update Your Key:**
   
   **Option A: Via `.env` file**
   ```env
   GROQ_API_KEY=your_new_key_here
   ```
   
   **Option B: Via Settings**
   - Go to Settings → Model
   - Enter new API key in "AI Enhancement"
   - Click "💾 Save Model Settings"
   - Restart the app

5. **Test the API Key:**
   - Go to Settings → Model
   - Click "🧪 Test API Key" button
   - Should show "✅ API key is valid and working!"

### Error: "401 Unauthorized"

**Cause:** Invalid or expired API key

**Solution:**
1. Get a new API key from [groq.com](https://groq.com)
2. Update in `.env` or Settings
3. Restart the app

### Error: "429 Rate Limit"

**Cause:** Too many requests in short time

**Solution:**
- Wait 1-2 minutes
- Try again
- Consider upgrading your Groq plan for higher limits

### Error: "Network error" or "Timeout"

**Cause:** Internet connection or Groq service issue

**Solution:**
1. Check your internet connection
2. Try again in a moment
3. Check Groq status: [status.groq.com](https://status.groq.com)

## Verification Steps

### 1. Check API Key Exists

**Windows:**
```cmd
type .env
```

**Linux/Mac:**
```bash
cat .env
```

Should show:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 2. Verify API Key Format

✅ **Correct:**
```env
GROQ_API_KEY=your-groq-api-key-here
```

❌ **Incorrect:**
```env
GROQ_API_KEY="gsk_..."  # No quotes
GROQ_API_KEY = gsk_...  # No spaces around =
GROQ_API_KEY=           # Empty
```

### 3. Test in Settings

1. Launch app: `streamlit run src\ui\app.py`
2. Go to Settings → Model
3. Check "AI Enhancement" section
4. Should show "✅ AI Analysis: Enabled"
5. Click "🧪 Test API Key"
6. Should show success message

### 4. Test with Image Upload

1. Go to AI Scanner
2. Upload a plant image
3. Wait for detection
4. Should see "🤖 AI-Powered Expert Analysis"
5. If error appears, check the message

## Quick Fix Checklist

- [ ] API key starts with `gsk_`
- [ ] No quotes in `.env` file
- [ ] No spaces around `=` in `.env`
- [ ] `.env` file exists in project root
- [ ] Restarted app after changing key
- [ ] Internet connection working
- [ ] Groq service is online

## Getting a New API Key

### Step-by-Step:

1. **Visit Groq:**
   - Go to [groq.com](https://groq.com)
   - Sign in (or create account)

2. **Navigate to API Keys:**
   - Click on your profile
   - Select "API Keys"

3. **Generate New Key:**
   - Click "Create API Key"
   - Give it a name (e.g., "AgroDetect AI")
   - Click "Generate"

4. **Copy the Key:**
   - Copy the entire key (starts with `gsk_`)
   - Save it securely

5. **Update in AgroDetect:**
   - Edit `.env` file
   - Replace old key with new key
   - Save file
   - Restart app

## Still Not Working?

### Check Groq Service Status

Visit: [status.groq.com](https://status.groq.com)

If Groq is down, wait for service to resume.

### Verify Model Name

In `src/utils/groq_analyzer.py`, check:
```python
self.model = "mixtral-8x7b-32768"
```

This should be a valid Groq model name.

### Check Request Format

The API request should look like:
```json
{
  "model": "mixtral-8x7b-32768",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert..."
    },
    {
      "role": "user",
      "content": "Disease info..."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Enable Debug Mode

To see detailed error messages:

1. Check terminal/console output
2. Look for full error traceback
3. Note the exact error message

## Alternative: Use Without AI

If AI analysis continues to fail:

1. **Standard remedies still work!**
   - Disease information shown
   - Symptoms listed
   - Organic remedies provided
   - Chemical treatments listed
   - Prevention strategies shown

2. **AI is optional enhancement**
   - Core functionality works without it
   - You get expert remedies from database
   - AI just adds contextual insights

## Contact Support

If none of these solutions work:

1. **Check Groq Documentation:**
   - [docs.groq.com](https://docs.groq.com)

2. **Groq Support:**
   - Contact Groq support for API issues

3. **AgroDetect Issues:**
   - Check project documentation
   - Review error messages carefully

## Summary

**Most Common Fix:**
1. Get new API key from groq.com
2. Update `.env` file: `GROQ_API_KEY=new_key_here`
3. Restart app
4. Test in Settings

**Remember:**
- Standard remedies always work
- AI is an enhancement, not required
- Check API key format carefully
- Restart app after changes

---

**Need Help?**
- See GROQ_API_SETUP.md for setup guide
- See GROQ_AI_INTEGRATION.md for full documentation
