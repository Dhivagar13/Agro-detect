# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub repository with the code
- Streamlit Cloud account (https://share.streamlit.io/)
- Groq API key
- Gemini API key

## Deployment Steps

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Update for Streamlit Cloud deployment"
git push origin main
```

### 2. Configure Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository: `Dhivagar13/Agro-detect`
4. Set main file path: `streamlit_app.py`
5. Click "Advanced settings"

### 3. Add Secrets

In the Streamlit Cloud dashboard, go to:
**App Settings → Secrets**

Add the following secrets:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
GEMINI_API_KEY = "your-gemini-api-key-here"
```

### 4. Python Version

Streamlit Cloud uses Python 3.13 by default. Our requirements.txt is compatible:
- `tensorflow>=2.16.0` (supports Python 3.13)
- `opencv-python-headless` (for cloud deployment)
- All other dependencies are compatible

### 5. Deploy

Click "Deploy" and wait for the app to build and start.

## Troubleshooting

### TensorFlow Installation Issues
If you see errors about TensorFlow compatibility:
- Ensure `requirements.txt` has `tensorflow>=2.16.0` (not 2.15.0)
- TensorFlow 2.16+ supports Python 3.13
- Clear Streamlit Cloud cache: Settings → Clear cache → Reboot

### API Key Issues
- Verify secrets are added in Streamlit Cloud dashboard
- Check secret names match exactly: `GROQ_API_KEY` and `GEMINI_API_KEY`
- Secrets are case-sensitive

### Model Loading Issues
- The app will prompt to upload a trained model on first run
- Or train a new model using the Training page
- Model files are not included in git (too large)

### Memory Issues
- Streamlit Cloud free tier has 1GB RAM limit
- MobileNetV2 is lightweight and should work fine
- If issues persist, consider upgrading to paid tier

## App URL
After deployment, your app will be available at:
`https://agro-detect-[random-id].streamlit.app/`

## Local Testing
Test locally before deploying:
```bash
streamlit run streamlit_app.py
```

## Environment Variables
The app reads from:
1. `.env` file (local development)
2. Streamlit secrets (cloud deployment)
3. Environment variables (fallback)

## Files Required for Deployment
- `streamlit_app.py` - Entry point
- `requirements.txt` - Dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `src/` - Application code
- `.env.example` - Template for environment variables

## Files NOT Deployed
- `.env` - Contains secrets (gitignored)
- `models/` - Model files (too large, upload via UI)
- `data/` - Dataset (too large)
- `logs/` - Log files
- `__pycache__/` - Python cache
