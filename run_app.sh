#!/bin/bash

# CKD Prediction Dashboard Launcher
# This script starts the Streamlit web application

echo "🏥 CKD Risk Prediction Dashboard"
echo "=================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p app/database
mkdir -p models
mkdir -p data/synthetic

echo "✅ Starting web application..."
echo ""
echo "📌 The dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run app/app.py --server.port 8501 --server.headless true
