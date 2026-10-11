#!/usr/bin/env bash
# Startup script for AQUASCAN 3D Drinking Water Potability Prediction
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  AQUASCAN 3D — Drinking Water Potability Prediction     "
echo "  Powered by Random Forest Classifier & Three.js 3D WebGL"
echo "=========================================================="

if [ ! -f "water_potability_rf.joblib" ]; then
    echo "[*] Training Random Forest Classifier..."
    python3 train_model.py
fi

echo "[*] Launching Web Application on http://localhost:5001"
python3 app.py
