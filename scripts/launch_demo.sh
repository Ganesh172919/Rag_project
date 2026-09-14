#!/bin/bash
# Launch AARAG demo application

echo "Launching AARAG Demo..."
echo "Open http://localhost:8501 in your browser"

streamlit run demo/app.py --server.port 8501 --server.headless true
