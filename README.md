# AQUASCAN 3D — Drinking Water Potability Prediction

An interactive, physics-driven 3D WebGL application paired with a **Random Forest Machine Learning model** to evaluate and visualize drinking water potability based on WHO/EPA international safety standards.

---

## 🌟 Key Features

### 1. 🤖 Random Forest Classifier
- **Ensemble Architecture**: 150 Decision Trees with bootstrap aggregation and Gini impurity splitting.
- **Trained on Multi-parameter Water Data**:
  - `pH Level` (Acidity / Alkalinity: 0 – 14)
  - `Hardness` (Calcium & Magnesium concentrations in mg/L)
  - `Total Dissolved Solids (TDS)` (Salts & minerals in ppm)
  - `Chloramines` (Disinfectant concentration in ppm)
  - `Sulfate` (Naturally occurring minerals in mg/L)
  - `Conductivity` (Electrical conductivity in μS/cm)
  - `Total Organic Carbon (TOC)` (Organic compound concentration in ppm)
  - `Trihalomethanes (THMs)` (Disinfection byproducts in μg/L)
  - `Turbidity` (Cloudiness / light scattering in NTU)
- **Model Performance**:
  - **Accuracy**: ~81.7%
  - **ROC-AUC**: ~88.3%
  - **F1-Score**: ~76.7%
  - **Precision**: ~76.4%

### 2. 🌊 Interactive 3D WebGL UI/UX (Three.js)
- **Mode 1: Laboratory Glass Beaker & Liquid Wave Simulation**:
  - Borosilicate glass beaker with graduation marks (100ml to 500ml).
  - Dynamic fluid surface waves using sine wave physics.
  - Realistic fluid color mapping:
    - Acidic ($pH < 6.0$): Warm amber/reddish glow.
    - Neutral ($6.5 \le pH \le 8.5$): Pristine azure cyan.
    - Alkaline ($pH > 8.5$): Deep indigo/violet.
    - Contaminated: Olive murky colloidal tint.
  - Effervescent micro-bubble particles responding to Chloramines and Trihalomethanes.
  - Floating suspended particulate matter scaling with Total Dissolved Solids and Turbidity.
  - Glowing LED pedestal that changes from Emerald Green (Potable) to Ruby Crimson (Unsafe).
- **Mode 2: 3D Molecular Simulation**:
  - Interactive Central $H_2O$ water molecule (Red Oxygen, White Hydrogen, covalent bonding).
  - Orbiting mineral and contaminant ions ($Ca^{2+}$, $Mg^{2+}$, $Cl^-$, $SO_4^{2-}$).
- **Mode 3: 3D WHO Radar Topology**:
  - 3D holographic quality polygon projecting compliance across all 9 parameters.
- **Audio Feedback**:
  - Pure Web Audio API procedural synthesis: water drop frequencies and verdict chimes (no audio files needed, fully offline).

### 3. 🔬 WHO Compliance & Prescriptive Treatment
- Real-time WHO benchmark validation for each parameter.
- **Auto-Optimize to Potable (What-If Analysis)**: One-click optimization calculating the exact minimal treatment adjustments needed to purify contaminated water.
- Water Quality Index (WQI) score calculation (0–100).
- Prescriptive engineering guidance (Reverse Osmosis, GAC Filtration, Aeration, Soda Ash Neutralization).

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- `flask`, `scikit-learn`, `pandas`, `numpy`, `joblib`

### 2. Run the App
```bash
cd /Users/haricharanreddygaddam/.gemini/antigravity/scratch/water-potability-3d
./start.sh
```
Or directly:
```bash
python3 app.py
```
Open your browser at:
👉 **[http://localhost:5001](http://localhost:5001)**

### 3. Run the Test Suite
```bash
python3 test_api.py
```

---

## 📁 Project Structure

```
water-potability-3d/
├── app.py                      # Flask backend API & simulation controller
├── train_model.py              # Random Forest training and metrics generator
├── test_api.py                 # Automated unit tests for APIs and model inference
├── start.sh                    # Launcher script
├── water_potability.csv        # Dataset containing 3,276 water samples
├── water_potability_rf.joblib  # Trained Random Forest scikit-learn model
├── model_metrics.json          # Persisted evaluation metrics & feature importances
├── templates/
│   └── index.html              # Modern 3D WebGL UI with Three.js & Tailwind
├── static/
│   ├── css/
│   └── js/
└── README.md
```
