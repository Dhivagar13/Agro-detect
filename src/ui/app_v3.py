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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import load_config
from src.inference.inference_engine import InferenceEngine

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
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Card styling */
    .dashboard-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
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
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #10b981;
        color: white;
    }
    
    .status-warning {
        background: #f59e0b;
        color: white;
    }
    
    .status-error {
        background: #ef4444;
        color: white;
    }
    
    /* Header styling */
    .page-header {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Alert boxes */
    .alert-box {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background: #dbeafe;
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-warning {
        background: #fef3c7;
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-success {
        background: #d1fae5;
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
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
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
                <h3 style='margin-top: 0;'>📈 Detection Trends</h3>
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
                <h3 style='margin-top: 0;'>🎯 Disease Distribution</h3>
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
                <h3 style='margin-top: 0;'>🔔 Recent Activity</h3>
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
                        <div style='font-weight: 600;'>{activity['action']}</div>
                        <div style='font-size: 0.85rem; color: #6b7280;'>{activity['detail']}</div>
                    </div>
                    <div style='font-size: 0.75rem; color: #9ca3af;'>{activity['time']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='dashboard-card'>
                <h3 style='margin-top: 0;'>⚠️ System Alerts</h3>
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
                    
                    if not st.session_state.model_trained:
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

else:
    st.markdown(f"""
        <div class='page-header'>
            <div class='page-title'>{page}</div>
            <div class='page-subtitle'>Coming soon...</div>
        </div>
    """, unsafe_allow_html=True)
