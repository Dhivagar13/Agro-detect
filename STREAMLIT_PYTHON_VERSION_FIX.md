# 🔧 CRITICAL: Change Python Version in Streamlit Cloud Dashboard

## ⚠️ Important Discovery

**Streamlit Cloud does NOT use `runtime.txt` to set Python version!**

The Python version must be changed in the Streamlit Cloud dashboard UI, not through files in the repository.

## 📋 How to Fix (Step-by-Step)

### 1. Go to Streamlit Cloud Dashboard
- Visit: https://share.streamlit.io/
- Log in to your account

### 2. Find Your App
- Look for: `agro-detect-qmpheriwpxxzbhruseeaot`
- Or search for: `agro-detect`

### 3. Open App Settings
- Click on your app
- Click the **⋮** (three dots menu) or **Settings** button
- Select **Settings** or **Advanced settings**

### 4. Change Python Version
Look for one of these options:
- **Python version** dropdown
- **Advanced settings** → **Python version**
- **App settings** → **Python version**

**Select: Python 3.11**

### 5. Save and Reboot
- Click **Save** or **Apply**
- Click **Reboot app**
- Wait 3-5 minutes for rebuild

## 🎯 What to Expect

After changing to Python 3.11:
```
Using Python 3.11.x environment ✓
Installing tensorflow==2.15.0 ✓
Installing opencv-python-headless==4.10.0.84 ✓
Installing streamlit==1.31.1 ✓
...
🎈 Your app is live!
```

## 📸 Visual Guide

The Python version setting is typically found in:
1. **App dashboard** → Click your app
2. **Settings** (gear icon or three dots menu)
3. **Advanced settings** or **General settings**
4. **Python version** dropdown

## ⚠️ Common Locations

Different Streamlit Cloud UI versions may have it in:
- Settings → Advanced → Python version
- Settings → General → Python version
- App settings → Runtime → Python version
- Deployment settings → Python version

## 🔍 If You Can't Find It

If you don't see a Python version option:
1. Try clicking **"Reboot app"** first
2. Look for **"Advanced settings"** or **"Show advanced options"**
3. Check the deployment logs - there might be a link to change settings
4. Contact Streamlit support if the option is missing

## 📝 Alternative: Redeploy

If changing settings doesn't work:
1. Delete the current deployment
2. Create a new deployment
3. During deployment setup, look for Python version selection
4. Choose Python 3.11
5. Complete deployment

## ✅ Verification

After changing Python version, check the deployment logs:
- Should show: `Using Python 3.11.x environment`
- NOT: `Using Python 3.13.12 environment`

## 🔑 Don't Forget API Keys

After successful deployment with Python 3.11:
1. Go to **Settings → Secrets**
2. Add:
```toml
GROQ_API_KEY = "your-groq-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
```
3. Save and reboot

## 📚 References

According to [Streamlit Community discussions](https://discuss.streamlit.io/t/deployment-failing-runtime-txt-is-ignored-forcing-python-3-13/114953):
> "Community Cloud does not use a runtime.txt file to set the version of Python. Instead, you need to select the version of Python in the UI (or change it from the app settings in your dashboard)."

## 🎯 Summary

1. ❌ `runtime.txt` does NOT work on Streamlit Cloud
2. ✅ Change Python version in dashboard UI
3. ✅ Select Python 3.11
4. ✅ Reboot app
5. ✅ Add API keys as secrets
6. ✅ App should deploy successfully

---

**Action Required**: Change Python version to 3.11 in Streamlit Cloud dashboard  
**Location**: App Settings → Python version  
**Expected Result**: Successful deployment with TensorFlow 2.15.0
