"""Main entry point for Streamlit deployment"""

# This file serves as the entry point for Streamlit Cloud and other deployment platforms
# It simply imports and runs the main application

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main app
from src.ui.app import *
