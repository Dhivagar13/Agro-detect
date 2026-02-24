"""Alternative entry point for Streamlit app"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main app
from src.ui.app import *
