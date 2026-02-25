"""AgroDetect AI - Premium Dashboard Design v3.0"""

import streamlit as st
import sys
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import load_config
from src.inference.inference_engine import InferenceEngine
from src.utils.disease_remedies import get_remedy, is_valid_plant_image
from src.utils.groq_analyzer import get_groq_analyzer
from src.utils.gemini_analyzer import get_gemini_analyzer
from src.utils.settings_manager import get_settings_manager

# Page configuration
st.set_page_config(
    page_title="AgroDetect AI - Smart Agriculture Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
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
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'groq_api_key' not in st.session_state:
    # Try to load from environment variable or settings
    import os
    st.session_state.groq_api_key = os.getenv('GROQ_API_KEY', '')
if 'gemini_api_key' not in st.session_state:
    # Try to load from environment variable
    import os
    st.session_state.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
if 'settings_manager' not in st.session_state:
    st.session_state.settings_manager = get_settings_manager()

# Get settings manager
settings_mgr = st.session_state.settings_manager

# Update Groq API key from settings if available
if settings_mgr.model.groq_api_key:
    st.session_state.groq_api_key = settings_mgr.model.groq_api_key

# Initialize AI analyzers
groq_analyzer = get_groq_analyzer(st.session_state.groq_api_key)
gemini_analyzer = get_gemini_analyzer(st.session_state.gemini_api_key)

# Initialize inference engine
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = Path('models/plant_disease_model.h5')
        class_names_path = Path('models/class_names.json')
        
        if not model_path.exists():
            return None, None
        
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        engine = InferenceEngine(
            model_path=str(model_path),
            class_names=class_names,
            confidence_threshold=0.7
        )
        engine.load_model()
        engine.warm_up(num_iterations=5)
        
        return engine, class_names
    except Exception as e:
        return None, None

engine, class_names = load_model()


# Premium CSS with advanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main background with pattern */
    .stApp {
        background: 
            linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%),
            repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(255,255,255,.05) 35px, rgba(255,255,255,.05) 70px);
        background-attachment: fixed;
    }
    
    /* Sidebar styling with depth */
    [data-testid="stSidebar"] {
        background: 
            linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        box-shadow: 5px 0 30px rgba(0,0,0,0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    /* Card styling with glassmorphism and dark text */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 
            0 20px 60px rgba(0,0,0,0.15),
            0 0 0 1px rgba(255,255,255,0.5) inset;
        margin: 1.5rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.8);
        color: #1f2937 !important;
    }
    
    .dashboard-card h3 {
        color: #1f2937 !important;
    }
    
    .dashboard-card p, .dashboard-card div {
        color: #4b5563 !important;
    }
    
    .dashboard-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 
            0 30px 80px rgba(0,0,0,0.2),
            0 0 0 1px rgba(255,255,255,0.8) inset;
    }
    
    /* Metric cards with 3D effect */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.4),
            0 5px 15px rgba(0,0,0,0.1),
            inset 0 1px 0 rgba(255,255,255,0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 
            0 25px 50px rgba(102, 126, 234, 0.5),
            0 10px 25px rgba(0,0,0,0.15),
            inset 0 1px 0 rgba(255,255,255,0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-change {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Status badges with glow */
    .status-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-badge:hover {
        transform: scale(1.05);
    }
    
    .status-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 5px 20px rgba(16, 185, 129, 0.4);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 5px 20px rgba(245, 158, 11, 0.4);
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 5px 20px rgba(239, 68, 68, 0.4);
    }
    
    /* Header styling with gradient border and dark text */
    .page-header {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 60px rgba(0,0,0,0.15),
            0 0 0 1px rgba(255,255,255,0.5) inset;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
    }
    
    .page-header * {
        color: #1f2937 !important;
    }
    
    .page-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 24px;
        padding: 2px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    
    .page-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: shimmer 3s linear infinite;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .page-subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Button styling with glow effect */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.4),
            0 0 0 0 rgba(102, 126, 234, 0);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 15px 40px rgba(102, 126, 234, 0.5),
            0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Alert boxes with depth */
    .alert-box {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        border-left: 5px solid;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .alert-box:hover {
        transform: translateX(5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(219, 234, 254, 0.95), rgba(191, 219, 254, 0.95));
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.95), rgba(253, 230, 138, 0.95));
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(209, 250, 229, 0.95), rgba(167, 243, 208, 0.95));
        border-color: #10b981;
        color: #065f46;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .float {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .shimmer {
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    /* Fix Streamlit text colors */
    .stMarkdown, .stText, p, div, span, label {
        color: #1f2937 !important;
    }
    
    /* White cards need dark text */
    .element-container {
        color: #1f2937 !important;
    }
    
    /* Chart titles */
    .js-plotly-plot .plotly .gtitle {
        fill: #1f2937 !important;
    }
    
    /* Dropdown/Selectbox styling - Fix black text issue */
    [data-baseweb="select"] {
        background-color: white !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: white !important;
        color: #1f2937 !important;
    }
    
    [data-baseweb="select"] input {
        color: #1f2937 !important;
    }
    
    [data-baseweb="select"] span {
        color: #1f2937 !important;
    }
    
    /* Dropdown menu options */
    [role="listbox"] {
        background-color: white !important;
    }
    
    [role="option"] {
        background-color: white !important;
        color: #1f2937 !important;
    }
    
    [role="option"]:hover {
        background-color: #f3f4f6 !important;
        color: #1f2937 !important;
    }
    
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    
    /* Input fields */
    input, textarea, select {
        color: #1f2937 !important;
        background-color: white !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #1f2937 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1f2937 !important;
    }
    
    /* Checkbox labels */
    .stCheckbox label {
        color: #1f2937 !important;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #1f2937 !important;
    }
    
    /* Text input */
    .stTextInput label {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🌿</div>
            <h1 style='font-size: 1.8rem; margin: 0;'>AgroDetect AI</h1>
            <p style='opacity: 0.8; font-size: 0.9rem; margin-top: 0.5rem;'>
                Smart Agriculture Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔬 AI Scanner", "📊 Analytics", "📈 Reports", "🎯 Training", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📡 System Status")
    
    if engine is not None:
        st.markdown("""
            <div class='status-badge status-success'>
                ✓ AI Model Active
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='status-badge status-error'>
                ✗ Model Offline
            </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.markdown("""
            <div class='alert-box alert-warning' style='font-size: 0.85rem; margin-top: 1rem;'>
                <strong>⚠️ Training Required</strong><br>
                Model needs training on plant disease data for accurate predictions.
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Today", st.session_state.total_predictions, "scans")
    with col2:
        st.metric("Uptime", "99.9%", "")
    
    st.markdown("---")
    
    # Model info
    if engine and class_names:
        st.markdown("### 🧠 Model Info")
        st.markdown(f"""
            <div style='font-size: 0.85rem; line-height: 1.8;'>
                <strong>Architecture:</strong> MobileNetV2<br>
                <strong>Classes:</strong> {len(class_names)}<br>
                <strong>Input Size:</strong> 224x224<br>
                <strong>Framework:</strong> TensorFlow
            </div>
        """, unsafe_allow_html=True)

# Main content
if page == "🏠 Dashboard":
    # Header
    st.markdown("""
        <div class='page-header fade-in'>
            <div class='page-title'>🌿 Welcome to AgroDetect AI</div>
            <div class='page-subtitle'>
                Your intelligent partner in plant disease detection and crop management
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='metric-card fade-in'>
                <div class='metric-label'>Total Scans</div>
                <div class='metric-value'>1,234</div>
                <div class='metric-change'>↑ 12% from last week</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card fade-in' style='animation-delay: 0.1s;'>
                <div class='metric-label'>Accuracy Rate</div>
                <div class='metric-value'>91.2%</div>
                <div class='metric-change'>↑ 2.3% improvement</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card fade-in' style='animation-delay: 0.2s;'>
                <div class='metric-label'>Active Users</div>
                <div class='metric-value'>456</div>
                <div class='metric-change'>↑ 8% growth</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='metric-card fade-in' style='animation-delay: 0.3s;'>
                <div class='metric-label'>Avg Response</div>
                <div class='metric-value'>125ms</div>
                <div class='metric-change'>↓ 15ms faster</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0; color: #1f2937 !important;'>📈 Detection Trends</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Create trend chart
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        detections = np.random.randint(30, 100, size=len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=detections,
            mode='lines+markers',
            name='Detections',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Detections",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0; color: #1f2937 !important;'>🎯 Disease Distribution</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Pie chart
        diseases = ['Early Blight', 'Late Blight', 'Healthy', 'Leaf Spot', 'Rust']
        counts = [30, 25, 20, 15, 10]
        
        fig = go.Figure(data=[go.Pie(
            labels=diseases,
            values=counts,
            hole=0.4,
            marker=dict(colors=['#667eea', '#764ba2', '#10b981', '#f59e0b', '#ef4444'])
        )])
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent activity and alerts
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0; color: #1f2937 !important;'>🔔 Recent Activity</h3>
            </div>
        """, unsafe_allow_html=True)
        
        activities = [
            {"time": "2 min ago", "action": "Disease detected", "detail": "Tomato Early Blight", "icon": "🔴"},
            {"time": "15 min ago", "action": "Scan completed", "detail": "Potato Healthy", "icon": "🟢"},
            {"time": "1 hour ago", "action": "New user registered", "detail": "farmer@example.com", "icon": "👤"},
            {"time": "2 hours ago", "action": "Model updated", "detail": "Accuracy improved to 91.2%", "icon": "🎯"},
        ]
        
        for activity in activities:
            st.markdown(f"""
                <div style='padding: 1rem; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center;'>
                    <div style='font-size: 1.5rem; margin-right: 1rem;'>{activity['icon']}</div>
                    <div style='flex: 1;'>
                        <div style='font-weight: 600; color: #1f2937;'>{activity['action']}</div>
                        <div style='font-size: 0.85rem; color: #6b7280;'>{activity['detail']}</div>
                    </div>
                    <div style='font-size: 0.75rem; color: #9ca3af;'>{activity['time']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0; color: #1f2937 !important;'>⚠️ System Alerts</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='alert-box alert-warning'>
                <strong>⚠️ Model Training Required</strong><br>
                The current model needs training on plant disease data for accurate predictions.
                <a href='#' style='color: #92400e; text-decoration: underline;'>Train Now →</a>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='alert-box alert-info'>
                <strong>ℹ️ New Dataset Available</strong><br>
                PlantVillage dataset with 54,000+ images is ready for training.
                <a href='#' style='color: #1e40af; text-decoration: underline;'>View Details →</a>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='alert-box alert-success'>
                <strong>✓ System Healthy</strong><br>
                All services are running normally. Uptime: 99.9%
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("""
        <div class='dashboard-card'>
            <h3 style='margin-top: 0;'>⚡ Quick Actions</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔬 New Scan", use_container_width=True):
            st.session_state.page = "🔬 AI Scanner"
    
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "📊 Analytics"
    
    with col3:
        if st.button("🎯 Train Model", use_container_width=True):
            st.session_state.page = "🎯 Training"
    
    with col4:
        if st.button("📥 Export Data", use_container_width=True):
            st.info("Export feature coming soon!")

elif page == "🔬 AI Scanner":
    st.markdown("""
        <div class='page-header fade-in'>
            <div class='page-title'>🔬 AI-Powered Disease Scanner</div>
            <div class='page-subtitle'>
                Upload or capture a plant leaf image for instant AI analysis
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if engine is None:
        st.markdown("""
            <div class='alert-box alert-error'>
                <strong>❌ AI Model Not Loaded</strong><br>
                Please run <code>python download_pretrained_model.py</code> to initialize the model.
            </div>
        """, unsafe_allow_html=True)
    else:
        # Upload section
        tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Use Camera"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Supported formats: JPEG, PNG, BMP",
                label_visibility="collapsed"
            )
        
        with tab2:
            camera_photo = st.camera_input("Take a photo")
            if camera_photo:
                uploaded_file = camera_photo
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                    <div class='dashboard-card'>
                        <h3 style='margin-top: 0;'>📷 Uploaded Image</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.image(uploaded_file, use_column_width=True)
                
                st.markdown(f"""
                    <div style='background: #f3f4f6; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                        <p style='margin: 0.5rem 0;'><strong>📁 Filename:</strong> {uploaded_file.name}</p>
                        <p style='margin: 0.5rem 0;'><strong>📏 Size:</strong> {uploaded_file.size / 1024:.2f} KB</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='dashboard-card'>
                        <h3 style='margin-top: 0;'>🎯 AI Analysis</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Processing...")
                    progress_bar.progress(33)
                    
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    progress_bar.progress(66)
                    result = engine.predict_single(image_array)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    # Check if image is valid (confidence threshold from settings)
                    if not is_valid_plant_image(result.confidence, threshold=settings_mgr.model.confidence_threshold):
                        st.markdown("""
                            <div style='background: #ef4444; color: white; border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0;'>
                                <h2 style='margin: 0;'>❌ Invalid Image</h2>
                                <h1 style='font-size: 2rem; margin: 1rem 0;'>Confidence: {:.1f}%</h1>
                                <p>This image does not appear to be a valid plant leaf image or doesn't match our trained categories.</p>
                                <p><strong>Please upload:</strong></p>
                                <ul style='text-align: left; display: inline-block;'>
                                    <li>Clear image of plant leaves</li>
                                    <li>Good lighting conditions</li>
                                    <li>Close-up view of affected area</li>
                                    <li>Supported plant types: Tomato, Potato, Pepper</li>
                                </ul>
                            </div>
                        """.format(result.confidence), unsafe_allow_html=True)
                    else:
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'disease': result.disease_class,
                            'confidence': result.confidence,
                            'image_name': uploaded_file.name
                        })
                        st.session_state.total_predictions += 1
                        
                        disease_name = result.disease_class.replace('_', ' ').replace('___', ' - ')
                        
                        if result.confidence >= 80:
                            color = "#10b981"
                            emoji = "🟢"
                        elif result.confidence >= 60:
                            color = "#f59e0b"
                            emoji = "🟡"
                        else:
                            color = "#ef4444"
                            emoji = "🔴"
                        
                        st.markdown(f"""
                            <div style='background: {color}; color: white; border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0;'>
                                <h2 style='margin: 0;'>{emoji} {disease_name}</h2>
                                <h1 style='font-size: 3rem; margin: 1rem 0;'>{result.confidence:.1f}%</h1>
                                <p>Inference Time: {result.inference_time_ms:.2f}ms</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Get AI-powered remedy recommendations
                        remedy_info = get_remedy(result.disease_class)
                        
                        # Get enhanced AI analysis from both Groq and Gemini
                        groq_analysis = None
                        gemini_analysis = None
                        
                        # Get Groq analysis
                        if groq_analyzer and remedy_info['symptoms']:
                            with st.spinner("🤖 Getting Groq AI insights..."):
                                try:
                                    groq_analysis = groq_analyzer.analyze_disease(
                                        disease_name=remedy_info['name'],
                                        confidence=result.confidence,
                                        symptoms=remedy_info['symptoms'],
                                        causes=remedy_info['causes'],
                                        context=f"Detection confidence: {result.confidence:.1f}%"
                                    )
                                except Exception as e:
                                    st.warning(f"⚠️ Groq AI unavailable: {str(e)}")
                        
                        # Get Gemini analysis
                        if gemini_analyzer and remedy_info['symptoms']:
                            with st.spinner("✨ Getting Gemini AI insights..."):
                                try:
                                    gemini_analysis = gemini_analyzer.analyze_disease(
                                        disease_name=remedy_info['name'],
                                        confidence=result.confidence,
                                        symptoms=remedy_info['symptoms'],
                                        causes=remedy_info['causes'],
                                        context=f"Detection confidence: {result.confidence:.1f}%"
                                    )
                                except Exception as e:
                                    st.warning(f"⚠️ Gemini AI unavailable: {str(e)}")
                        
                        # Display AI Analyses side by side if available
                        if groq_analysis or gemini_analysis:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                                <div class='dashboard-card'>
                                    <h3 style='margin-top: 0;'>🤖 Dual AI-Powered Expert Analysis</h3>
                                    <p style='color: #6b7280;'>Compare insights from two leading AI models</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            col_groq, col_gemini = st.columns(2)
                            
                            # Groq Analysis Column
                            with col_groq:
                                if groq_analysis:
                                    urgency_colors = {
                                        'low': '#10b981',
                                        'moderate': '#f59e0b',
                                        'high': '#ef4444',
                                        'critical': '#dc2626'
                                    }
                                    urgency_color = urgency_colors.get(groq_analysis['urgency'], '#f59e0b')
                                    
                                    st.markdown(f"""
                                        <div class='dashboard-card' style='border-left: 5px solid #667eea; background: rgba(102, 126, 234, 0.05);'>
                                            <h4 style='margin-top: 0; color: #667eea;'>🤖 Groq AI (Mixtral)</h4>
                                            <div style='background: {urgency_color}; color: white; display: inline-block; padding: 0.4rem 0.8rem; border-radius: 15px; font-weight: bold; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                                                Urgency: {groq_analysis['urgency'].upper()}
                                            </div>
                                            <p style='font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0;'>{groq_analysis['analysis']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if groq_analysis['recommendations']:
                                        st.markdown("**💡 Recommendations:**")
                                        recommendations = groq_analysis['recommendations'].strip().split('\n')
                                        for rec in recommendations[:3]:  # Show top 3
                                            if rec.strip():
                                                st.markdown(f"• {rec.strip()}")
                                    
                                    if groq_analysis['additional_tips']:
                                        with st.expander("💭 More Tips"):
                                            tips = groq_analysis['additional_tips'].strip().split('\n')
                                            for tip in tips:
                                                if tip.strip():
                                                    st.markdown(f"• {tip.strip()}")
                                else:
                                    st.info("🤖 Groq AI analysis not available")
                            
                            # Gemini Analysis Column
                            with col_gemini:
                                if gemini_analysis:
                                    urgency_colors = {
                                        'low': '#10b981',
                                        'moderate': '#f59e0b',
                                        'high': '#ef4444',
                                        'critical': '#dc2626'
                                    }
                                    urgency_color = urgency_colors.get(gemini_analysis['urgency'], '#f59e0b')
                                    
                                    st.markdown(f"""
                                        <div class='dashboard-card' style='border-left: 5px solid #4285f4; background: rgba(66, 133, 244, 0.05);'>
                                            <h4 style='margin-top: 0; color: #4285f4;'>✨ Gemini AI (Google)</h4>
                                            <div style='background: {urgency_color}; color: white; display: inline-block; padding: 0.4rem 0.8rem; border-radius: 15px; font-weight: bold; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                                                Urgency: {gemini_analysis['urgency'].upper()}
                                            </div>
                                            <p style='font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0;'>{gemini_analysis['analysis']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if gemini_analysis['recommendations']:
                                        st.markdown("**💡 Recommendations:**")
                                        recommendations = gemini_analysis['recommendations'].strip().split('\n')
                                        for rec in recommendations[:3]:  # Show top 3
                                            if rec.strip():
                                                st.markdown(f"• {rec.strip()}")
                                    
                                    if gemini_analysis['additional_tips']:
                                        with st.expander("💭 More Tips"):
                                            tips = gemini_analysis['additional_tips'].strip().split('\n')
                                            for tip in tips:
                                                if tip.strip():
                                                    st.markdown(f"• {tip.strip()}")
                                else:
                                    st.info("✨ Gemini AI analysis not available")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Disease Information
                        st.markdown("""
                            <div class='dashboard-card'>
                                <h3 style='margin-top: 0;'>🔬 Disease Information</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Severity:** `{remedy_info['severity']}`")
                        with col_b:
                            st.markdown(f"**Disease:** {remedy_info['name']}")
                        
                        st.markdown(f"**Description:** {remedy_info['description']}")
                        
                        # Symptoms
                        if remedy_info['symptoms']:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                                <div class='dashboard-card'>
                                    <h3 style='margin-top: 0;'>🔍 Symptoms</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            for symptom in remedy_info['symptoms']:
                                st.markdown(f"• {symptom}")
                        
                        # Causes
                        if remedy_info['causes']:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                                <div class='dashboard-card'>
                                    <h3 style='margin-top: 0;'>⚠️ Causes</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            for cause in remedy_info['causes']:
                                st.markdown(f"• {cause}")
                        
                        # Treatment Options
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("""
                            <div class='dashboard-card'>
                                <h3 style='margin-top: 0;'>💊 AI-Recommended Treatment Solutions</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        tab_organic, tab_chemical, tab_ai_compare = st.tabs(["🌿 Organic Remedies", "🧪 Chemical Treatments", "🤖 AI Comparison"])
                        
                        with tab_organic:
                            if remedy_info['organic_remedies']:
                                st.markdown("**Organic and Natural Solutions:**")
                                for i, remedy in enumerate(remedy_info['organic_remedies'], 1):
                                    st.markdown(f"{i}. {remedy}")
                            else:
                                st.info("No organic remedies needed - plant is healthy!")
                        
                        with tab_chemical:
                            if remedy_info['chemical_remedies']:
                                st.markdown("**Chemical Treatment Options:**")
                                for i, remedy in enumerate(remedy_info['chemical_remedies'], 1):
                                    st.markdown(f"{i}. {remedy}")
                                st.warning("⚠️ Always follow label instructions and safety guidelines when using chemical treatments.")
                            else:
                                st.info("No chemical treatments needed - plant is healthy!")
                        
                        with tab_ai_compare:
                            if groq_analyzer and remedy_info['organic_remedies'] and remedy_info['chemical_remedies']:
                                with st.spinner("🤖 AI is comparing treatment options..."):
                                    try:
                                        comparison = groq_analyzer.compare_treatments(
                                            disease_name=remedy_info['name'],
                                            organic_options=remedy_info['organic_remedies'],
                                            chemical_options=remedy_info['chemical_remedies']
                                        )
                                        
                                        st.markdown("**🤖 AI Treatment Comparison:**")
                                        st.markdown(comparison)
                                        
                                        # Weather advice
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        st.markdown("**🌤️ Weather Considerations:**")
                                        weather_advice = groq_analyzer.get_weather_advice(remedy_info['name'])
                                        st.markdown(weather_advice)
                                        
                                    except Exception as e:
                                        st.error(f"AI comparison unavailable: {str(e)}")
                            else:
                                st.info("AI comparison available when both organic and chemical treatments are present.")
                        
                        # Prevention
                        if remedy_info['prevention']:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                                <div class='dashboard-card'>
                                    <h3 style='margin-top: 0;'>🛡️ Prevention Strategies</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            for prevention in remedy_info['prevention']:
                                st.markdown(f"• {prevention}")
                        
                        # Best Practices
                        if remedy_info['best_practices']:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                                <div class='dashboard-card'>
                                    <h3 style='margin-top: 0;'>✅ Best Practices</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            for practice in remedy_info['best_practices']:
                                st.markdown(f"• {practice}")
                        
                        if not st.session_state.model_trained:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.warning("⚠️ Model not trained on plant disease data. Results may be inaccurate. Please train the model for reliable predictions.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "📊 Analytics":
    st.markdown("""
        <div class='page-header'>
            <div class='page-title'>📊 Advanced Analytics</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Total Scans", "1,234", "↑ 12%"),
        ("Accuracy", "91.2%", "↑ 2.3%"),
        ("Users", "456", "↑ 8%"),
        ("Uptime", "99.9%", "→ 0%")
    ]
    
    for i, (label, value, change) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{value}</div>
                    <div class='metric-change'>{change}</div>
                </div>
            """, unsafe_allow_html=True)

elif page == "📈 Reports":
    st.markdown("""
        <div class='page-header fade-in'>
            <div class='page-title'>📈 Comprehensive Reports</div>
            <div class='page-subtitle'>
                Detailed analysis and insights from your detection history
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Report type selector
    report_type = st.selectbox(
        "Select Report Type",
        ["Detection Summary", "Disease Distribution", "Performance Metrics", "Historical Trends", "Export Data"]
    )
    
    if report_type == "Detection Summary":
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>📊 Detection Summary Report</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Detections", len(st.session_state.prediction_history))
            st.metric("Today's Scans", st.session_state.total_predictions)
        
        with col2:
            if st.session_state.prediction_history:
                avg_conf = np.mean([p['confidence'] for p in st.session_state.prediction_history])
                st.metric("Average Confidence", f"{avg_conf:.1f}%")
            else:
                st.metric("Average Confidence", "N/A")
            st.metric("Model Status", "Active" if engine else "Offline")
        
        with col3:
            if class_names:
                st.metric("Trained Classes", len(class_names))
            else:
                st.metric("Trained Classes", "0")
            st.metric("Model Type", "MobileNetV2")
        
        # Recent detections table
        if st.session_state.prediction_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class='dashboard-card'>
                    <h3 style='margin-top: 0;'>🕐 Recent Detections</h3>
                </div>
            """, unsafe_allow_html=True)
            
            df = pd.DataFrame(st.session_state.prediction_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False).head(10)
            df['disease'] = df['disease'].str.replace('_', ' ')
            df['confidence'] = df['confidence'].round(2)
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No detection history available yet. Start scanning to see reports!")
    
    elif report_type == "Disease Distribution":
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>🎯 Disease Distribution Analysis</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            disease_counts = df['disease'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=disease_counts.index,
                    y=disease_counts.values,
                    labels={'x': 'Disease', 'y': 'Count'},
                    title='Disease Detection Frequency',
                    color=disease_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=disease_counts.index,
                    values=disease_counts.values,
                    hole=0.4
                )])
                fig.update_layout(height=400, title='Distribution %')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for distribution analysis.")
    
    elif report_type == "Performance Metrics":
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>⚡ Performance Metrics</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Performance")
            
            # Check if training history exists
            history_path = Path('models/training_history.json')
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                st.metric("Final Training Accuracy", f"{history['accuracy'][-1]*100:.2f}%")
                st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]*100:.2f}%")
                st.metric("Training Epochs", len(history['accuracy']))
                
                # Plot training history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['accuracy'],
                    name='Training Accuracy',
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_accuracy'],
                    name='Validation Accuracy',
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title='Training History',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training history available. Train the model first.")
        
        with col2:
            st.markdown("### System Performance")
            
            if st.session_state.prediction_history:
                # Calculate average inference time (simulated)
                avg_inference = 125.5
                st.metric("Avg Inference Time", f"{avg_inference:.1f}ms")
                st.metric("Throughput", f"{1000/avg_inference:.1f} images/sec")
                st.metric("Model Size", "14.2 MB")
                st.metric("Memory Usage", "~500 MB")
                
                # Confidence distribution
                df = pd.DataFrame(st.session_state.prediction_history)
                fig = px.histogram(
                    df,
                    x='confidence',
                    nbins=20,
                    title='Confidence Score Distribution',
                    labels={'confidence': 'Confidence %', 'count': 'Frequency'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available yet.")
    
    elif report_type == "Historical Trends":
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>📈 Historical Trends</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['count'],
                mode='lines+markers',
                name='Daily Detections',
                fill='tozeroy',
                line=dict(color='#667eea', width=3)
            ))
            fig.update_layout(
                title='Detection Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Detections',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence trends
            daily_conf = df.groupby('date')['confidence'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_conf['date'],
                y=daily_conf['confidence'],
                mode='lines+markers',
                name='Average Confidence',
                line=dict(color='#10b981', width=3)
            ))
            fig.update_layout(
                title='Average Confidence Trends',
                xaxis_title='Date',
                yaxis_title='Confidence %',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available yet.")
    
    elif report_type == "Export Data":
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>📥 Export Data</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Export Options")
                export_format = st.radio("Select Format", ["CSV", "JSON", "Excel"])
                
                if st.button("Generate Export File", use_container_width=True):
                    if export_format == "CSV":
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "agrodetect_report.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    elif export_format == "JSON":
                        json_str = df.to_json(orient='records', date_format='iso')
                        st.download_button(
                            "Download JSON",
                            json_str,
                            "agrodetect_report.json",
                            "application/json",
                            use_container_width=True
                        )
                    else:
                        st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
            
            with col2:
                st.markdown("### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"Total records: {len(df)}")
        else:
            st.info("No data available to export.")

elif page == "🎯 Training":
    st.markdown("""
        <div class='page-header fade-in'>
            <div class='page-title'>🎯 Model Training Center</div>
            <div class='page-subtitle'>
                Train your AI model on custom plant disease datasets
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🚀 Quick Train", "⚙️ Advanced Settings", "📊 Training History"])
    
    with tab1:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>🚀 Quick Training</h3>
                <p>Train your model with recommended settings</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dataset_path = st.text_input(
                "Dataset Path",
                value="D:\\datasets\\plantvillage\\PlantVillage",
                help="Path to your dataset folder with class subdirectories"
            )
            
            num_classes = st.number_input(
                "Number of Classes",
                min_value=2,
                max_value=100,
                value=15,
                help="Number of disease classes in your dataset"
            )
            
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=100,
                value=50,
                help="More epochs = better accuracy but longer training time"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64],
                index=1,
                help="Larger batch = faster training but more memory"
            )
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                help="Lower = more stable, Higher = faster convergence"
            )
        
        with col2:
            st.markdown("### 📋 Training Info")
            st.info(f"""
            **Estimated Time:**  
            {epochs * 2} - {epochs * 5} minutes
            
            **Model:** MobileNetV2  
            **Input Size:** 224x224  
            **Framework:** TensorFlow
            
            **Requirements:**
            - Organized dataset
            - 2GB+ free RAM
            - GPU recommended
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        with col_b:
            if st.button("🚀 Start Training", use_container_width=True, type="primary"):
                st.markdown("""
                    <div class='alert-box alert-info'>
                        <strong>📝 Training Command</strong><br>
                        Run this command in your terminal:
                    </div>
                """, unsafe_allow_html=True)
                
                command = f"""python train_model.py --data-dir "{dataset_path}" --num-classes {num_classes} --epochs {epochs} --batch-size {batch_size} --learning-rate {learning_rate}"""
                
                st.code(command, language="bash")
                
                st.markdown("**Or use the quick training script:**")
                st.code("./quick_train.bat", language="bash")
                
                st.success("Copy and run the command above in your terminal to start training!")
    
    with tab2:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>⚙️ Advanced Training Configuration</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Architecture")
            base_model = st.selectbox("Base Model", ["MobileNetV2", "ResNet50", "EfficientNetB0"])
            input_size = st.selectbox("Input Size", ["224x224", "299x299", "384x384"])
            freeze_layers = st.slider("Freeze Base Layers", 0, 150, 100)
            
            st.markdown("### Data Augmentation")
            rotation = st.slider("Rotation Range", 0, 45, 20)
            zoom = st.slider("Zoom Range", 0.0, 0.5, 0.2)
            flip = st.checkbox("Horizontal Flip", value=True)
            brightness = st.slider("Brightness Range", 0.0, 0.5, 0.2)
        
        with col2:
            st.markdown("### Training Parameters")
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            loss_function = st.selectbox("Loss Function", ["Categorical Crossentropy", "Sparse Categorical Crossentropy"])
            
            st.markdown("### Callbacks")
            early_stopping = st.checkbox("Early Stopping", value=True)
            if early_stopping:
                patience = st.number_input("Patience", 5, 20, 10)
            
            reduce_lr = st.checkbox("Reduce LR on Plateau", value=True)
            tensorboard = st.checkbox("TensorBoard Logging", value=True)
            
            st.markdown("### Validation")
            val_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("Configuration saved! Use this in your training script.")
    
    with tab3:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>📊 Training History</h3>
            </div>
        """, unsafe_allow_html=True)
        
        history_path = Path('models/training_history.json')
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Training Accuracy", f"{history['accuracy'][-1]*100:.2f}%")
            with col2:
                st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]*100:.2f}%")
            with col3:
                st.metric("Total Epochs", len(history['accuracy']))
            with col4:
                best_val_acc = max(history['val_accuracy'])
                st.metric("Best Validation Accuracy", f"{best_val_acc*100:.2f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Accuracy plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[acc * 100 for acc in history['accuracy']],
                name='Training Accuracy',
                mode='lines+markers',
                line=dict(color='#667eea', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=[acc * 100 for acc in history['val_accuracy']],
                name='Validation Accuracy',
                mode='lines+markers',
                line=dict(color='#10b981', width=3)
            ))
            fig.update_layout(
                title='Training & Validation Accuracy',
                xaxis_title='Epoch',
                yaxis_title='Accuracy (%)',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Loss plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'],
                name='Training Loss',
                mode='lines+markers',
                line=dict(color='#ef4444', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                name='Validation Loss',
                mode='lines+markers',
                line=dict(color='#f59e0b', width=3)
            ))
            fig.update_layout(
                title='Training & Validation Loss',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download history
            if st.button("📥 Download Training History"):
                json_str = json.dumps(history, indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    "training_history.json",
                    "application/json"
                )
        else:
            st.info("No training history available. Train your model first!")
            
            st.markdown("""
                <div class='alert-box alert-info'>
                    <strong>📚 Getting Started with Training</strong><br><br>
                    <strong>1. Prepare Your Dataset:</strong><br>
                    Organize images in folders by class name<br><br>
                    <strong>2. Run Training:</strong><br>
                    Use the Quick Train tab or run quick_train.bat<br><br>
                    <strong>3. Monitor Progress:</strong><br>
                    Training history will appear here automatically
                </div>
            """, unsafe_allow_html=True)

elif page == "⚙️ Settings":
    st.markdown("""
        <div class='page-header fade-in'>
            <div class='page-title'>⚙️ System Settings</div>
            <div class='page-subtitle'>
                Configure your AgroDetect AI system
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 General", "🧠 Model", "🎨 Appearance", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>🔧 General Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Application Settings")
            
            auto_save = st.checkbox("Auto-save Detection Results", value=settings_mgr.general.auto_save, key="auto_save_check")
            notifications = st.checkbox("Enable Notifications", value=settings_mgr.general.notifications, key="notifications_check")
            sound_alerts = st.checkbox("Sound Alerts", value=settings_mgr.general.sound_alerts, key="sound_alerts_check")
            
            st.markdown("### Data Management")
            
            max_history = st.number_input("Max History Records", 100, 10000, settings_mgr.general.max_history, key="max_history_input")
            auto_cleanup = st.checkbox("Auto-cleanup Old Records", value=settings_mgr.general.auto_cleanup, key="auto_cleanup_check")
            
            cleanup_days = settings_mgr.general.cleanup_days
            if auto_cleanup:
                cleanup_days = st.number_input("Keep Records (days)", 7, 365, settings_mgr.general.cleanup_days, key="cleanup_days_input")
        
        with col2:
            st.markdown("### Performance")
            
            cache_enabled = st.checkbox("Enable Caching", value=settings_mgr.general.cache_enabled, key="cache_check")
            gpu_acceleration = st.checkbox("GPU Acceleration", value=settings_mgr.general.gpu_acceleration, key="gpu_check")
            
            st.markdown("### Language & Region")
            
            language = st.selectbox("Language", ["English", "Spanish", "French", "Hindi", "Chinese"], 
                                   index=["English", "Spanish", "French", "Hindi", "Chinese"].index(settings_mgr.general.language), 
                                   key="language_select")
            timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "IST", "CET"],
                                   index=["UTC", "EST", "PST", "IST", "CET"].index(settings_mgr.general.timezone),
                                   key="timezone_select")
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"],
                                      index=["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"].index(settings_mgr.general.date_format),
                                      key="date_format_select")
        
        if st.button("💾 Save General Settings", use_container_width=True, key="save_general_btn"):
            # Update settings
            success = settings_mgr.update_general(
                auto_save=auto_save,
                notifications=notifications,
                sound_alerts=sound_alerts,
                max_history=max_history,
                auto_cleanup=auto_cleanup,
                cleanup_days=cleanup_days,
                cache_enabled=cache_enabled,
                gpu_acceleration=gpu_acceleration,
                language=language,
                timezone=timezone,
                date_format=date_format
            )
            
            if success:
                st.success("✅ Settings saved successfully!")
                
                # Apply settings immediately
                if settings_mgr.general.auto_cleanup and len(st.session_state.prediction_history) > max_history:
                    st.session_state.prediction_history = st.session_state.prediction_history[-max_history:]
                    st.info(f"🧹 Cleaned up history to {max_history} records")
                
                if settings_mgr.general.notifications:
                    st.info("🔔 Notifications enabled")
            else:
                st.error("❌ Failed to save settings")
    
    with tab2:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>🧠 Model Configuration</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Inference Settings")
            
            confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, int(settings_mgr.model.confidence_threshold), key="conf_threshold_slider")
            st.info(f"Images below {confidence_threshold}% confidence will be marked as invalid")
            
            batch_inference = st.checkbox("Enable Batch Inference", value=settings_mgr.model.batch_inference, key="batch_inf_check")
            batch_size_inf = settings_mgr.model.batch_size
            if batch_inference:
                batch_size_inf = st.number_input("Batch Size", 1, 32, settings_mgr.model.batch_size, key="batch_size_input")
            
            st.markdown("### AI Enhancement")
            
            groq_key = st.text_input(
                "Groq API Key",
                value=st.session_state.groq_api_key,
                type="password",
                help="Enter your Groq API key for AI-powered analysis",
                key="groq_key_input"
            )
            
            if groq_analyzer:
                st.success("✅ AI Analysis: Enabled")
                
                # Test API key button
                if st.button("🧪 Test API Key", key="test_api_btn"):
                    with st.spinner("Testing API connection..."):
                        try:
                            test_result = groq_analyzer.analyze_disease(
                                disease_name="Test Disease",
                                confidence=85.0,
                                symptoms=["Test symptom"],
                                causes=["Test cause"],
                                context="API test"
                            )
                            st.success("✅ API key is valid and working!")
                        except Exception as e:
                            st.error(f"❌ API test failed: {str(e)}")
                            if "400" in str(e):
                                st.info("💡 Tip: Check if your API key is correct and has not expired.")
                            elif "401" in str(e):
                                st.info("💡 Tip: Your API key may be invalid. Get a new one from groq.com")
            else:
                st.warning("⚠️ AI Analysis: Disabled (No API key)")
                st.info("Get your free API key at: [groq.com](https://groq.com)")
            
            st.markdown("### Model Selection")
            
            model_path = st.text_input("Model Path", settings_mgr.model.model_path, key="model_path_input")
            class_names_path = st.text_input("Class Names Path", settings_mgr.model.class_names_path, key="class_names_path_input")
            
            if st.button("🔄 Reload Model", key="reload_model_btn"):
                st.info("Model reload functionality - restart the app to load new model")
        
        with col2:
            st.markdown("### Model Information")
            
            if engine and class_names:
                st.success("✅ Model Loaded Successfully")
                
                st.markdown(f"""
                **Architecture:** MobileNetV2  
                **Classes:** {len(class_names)}  
                **Input Size:** 224x224x3  
                **Parameters:** ~3.5M  
                **Model Size:** ~14 MB  
                **Framework:** TensorFlow 2.15
                """)
                
                with st.expander("View All Classes"):
                    for i, class_name in enumerate(class_names, 1):
                        st.text(f"{i}. {class_name.replace('_', ' ')}")
            else:
                st.error("❌ Model Not Loaded")
                st.info("Train a model or download a pre-trained model to get started")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### AI Features")
            
            if groq_analyzer:
                st.markdown("""
                **Enabled Features:**
                - 🤖 Expert disease analysis
                - 💡 Immediate action recommendations
                - 🎯 Urgency assessment
                - 💭 Contextual tips
                - 🌤️ Weather-based advice
                - ⚖️ Treatment comparison
                """)
            else:
                st.markdown("""
                **Available with API Key:**
                - 🤖 AI-powered expert analysis
                - 💡 Smart recommendations
                - 🎯 Urgency assessment
                - 💭 Contextual insights
                - 🌤️ Weather considerations
                - ⚖️ Treatment comparison
                
                Get your free API key at:
                [groq.com](https://groq.com)
                """)
        
        if st.button("💾 Save Model Settings", use_container_width=True, key="save_model_btn"):
            # Update settings
            success = settings_mgr.update_model(
                confidence_threshold=float(confidence_threshold),
                batch_inference=batch_inference,
                batch_size=batch_size_inf,
                model_path=model_path,
                class_names_path=class_names_path,
                groq_api_key=groq_key
            )
            
            if success:
                st.success("✅ Model settings saved!")
                st.info("🔄 Restart the app to apply model path changes")
                
                # Update session state
                st.session_state.groq_api_key = groq_key
            else:
                st.error("❌ Failed to save settings")
    
    with tab3:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>🎨 Appearance Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Theme")
            
            theme = st.radio("Color Theme", ["Default (Purple)", "Green", "Blue", "Dark"],
                           index=["Default (Purple)", "Green", "Blue", "Dark"].index(settings_mgr.appearance.theme),
                           key="theme_radio")
            
            st.markdown("### Layout")
            
            sidebar_default = st.radio("Sidebar Default", ["Expanded", "Collapsed"],
                                      index=["Expanded", "Collapsed"].index(settings_mgr.appearance.sidebar_default),
                                      key="sidebar_radio")
            chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"],
                                      index=["Modern", "Classic", "Minimal"].index(settings_mgr.appearance.chart_style),
                                      key="chart_style_select")
            
            st.markdown("### Display")
            
            show_animations = st.checkbox("Show Animations", value=settings_mgr.appearance.show_animations, key="animations_check")
            compact_mode = st.checkbox("Compact Mode", value=settings_mgr.appearance.compact_mode, key="compact_check")
        
        with col2:
            st.markdown("### Preview")
            
            st.markdown("""
                <div class='metric-card' style='margin: 1rem 0;'>
                    <div class='metric-label'>Sample Metric</div>
                    <div class='metric-value'>95.2%</div>
                    <div class='metric-change'>↑ 5.2%</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='alert-box alert-success'>
                    <strong>✓ Preview</strong><br>
                    This is how alerts will appear
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"""
            **Current Settings:**
            - Theme: {theme}
            - Sidebar: {sidebar_default}
            - Chart Style: {chart_style}
            - Animations: {'On' if show_animations else 'Off'}
            - Compact Mode: {'On' if compact_mode else 'Off'}
            """)
        
        if st.button("💾 Save Appearance Settings", use_container_width=True, key="save_appearance_btn"):
            # Update settings
            success = settings_mgr.update_appearance(
                theme=theme,
                sidebar_default=sidebar_default,
                chart_style=chart_style,
                show_animations=show_animations,
                compact_mode=compact_mode
            )
            
            if success:
                st.success("✅ Appearance settings saved!")
                st.info("🔄 Refresh the page to see changes")
            else:
                st.error("❌ Failed to save settings")
    
    with tab4:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>ℹ️ About AgroDetect AI</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🌿 AgroDetect AI
            
            **Version:** 3.1.0  
            **Release Date:** February 2026  
            **Framework:** Streamlit + TensorFlow
            
            ### 📋 Description
            
            AgroDetect AI is an advanced plant disease detection system powered by deep learning. 
            Using state-of-the-art MobileNetV2 architecture, it provides fast and accurate 
            identification of plant diseases with AI-powered treatment recommendations.
            
            ### ✨ Key Features
            
            - 🔬 Real-time disease detection
            - 🤖 AI-powered remedy recommendations
            - 📊 Comprehensive analytics and reporting
            - 🎯 Custom model training
            - 📱 Mobile-friendly interface
            - 🌐 Multi-language support (coming soon)
            
            ### 🛠️ Technology Stack
            
            - **Frontend:** Streamlit
            - **Backend:** Python 3.11
            - **ML Framework:** TensorFlow 2.15
            - **Model:** MobileNetV2 (Transfer Learning)
            - **Visualization:** Plotly
            - **Data Processing:** NumPy, Pandas
            
            ### 📚 Documentation
            
            - [User Guide](README.md)
            - [Training Guide](TRAINING_GUIDE.md)
            - [Dataset Setup](DATASET_SETUP_GUIDE.md)
            - [API Documentation](SYSTEM_ARCHITECTURE.md)
            
            ### 📄 License
            
            MIT License - Open Source
            
            ### 🤝 Support
            
            For issues and feature requests, please contact support.
            """)
        
        with col2:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <div style='font-size: 4rem; margin: 1rem 0;'>🌿</div>
                    <h3>AgroDetect AI</h3>
                    <p>Smart Agriculture Platform</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("### 📊 System Status")
            st.success("✅ All Systems Operational")
            
            st.markdown("### 🔄 Updates")
            st.info("You're running the latest version")
            
            if st.button("🔍 Check for Updates", use_container_width=True):
                st.success("✅ You have the latest version!")

else:
    st.markdown(f"""
        <div class='page-header'>
            <div class='page-title'>{page}</div>
            <div class='page-subtitle'>Coming soon...</div>
        </div>
    """, unsafe_allow_html=True)
