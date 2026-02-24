"""AgroDetect AI - Modern UI with Advanced Features"""

import streamlit as st
import sys
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import load_config
from src.inference.inference_engine import InferenceEngine
from PIL import Image

# Page configuration with custom theme
st.set_page_config(
    page_title="AgroDetect AI - Smart Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/agrodetect-ai',
        'Report a bug': "https://github.com/yourusername/agrodetect-ai/issues",
        'About': "AgroDetect AI - Powered by Deep Learning"
    }
)

# Load configuration
try:
    config = load_config()
except Exception as e:
    config = {}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize inference engine
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = Path('models/plant_disease_model.h5')
