#!/bin/bash
# Setup script for AgroDetect AI on Linux/Mac

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the Streamlit app:"
echo "  streamlit run src/ui/app.py"
echo ""
echo "To run tests:"
echo "  pytest"
