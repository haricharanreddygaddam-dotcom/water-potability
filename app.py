"""
Flask Application for Drinking Water Potability Prediction
Provides 3D Simulation Engine APIs and Machine Learning Inference
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "water_potability_rf.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# Load model and metrics
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        model_metrics = json.load(f)
else:
    model_metrics = {}

# WHO Guidelines and feature metadata
FEATURE_CONFIG = {
    "ph": {
        "name": "pH Level",
        "unit": "pH",
        "min": 0.0,
        "max": 14.0,
        "default": 7.2,
        "step": 0.1,
        "who_min": 6.5,
        "who_max": 8.5,
        "ideal": 7.3,
        "category": "Chemical",
        "description": "Acidity / alkalinity scale. WHO benchmark: 6.5 – 8.5."
    },
    "Hardness": {
        "name": "Hardness",
        "unit": "mg/L",
        "min": 40.0,
        "max": 340.0,
        "default": 185.0,
        "step": 1.0,
        "who_min": 60.0,
        "who_max": 200.0,
        "ideal": 120.0,
        "category": "Minerals",
        "description": "Dissolved calcium & magnesium ions. Safe limit < 200 mg/L."
    },
    "Solids": {
        "name": "Total Dissolved Solids (TDS)",
        "unit": "ppm",
        "min": 300.0,
        "max": 55000.0,
        "default": 14500.0,
        "step": 100.0,
        "who_min": 100.0,
        "who_max": 1000.0,
        "ideal": 300.0,
        "category": "Physical",
        "description": "Total dissolved salts and minerals. Palatable < 500 ppm."
    },
    "Chloramines": {
        "name": "Chloramines",
        "unit": "ppm",
        "min": 0.3,
        "max": 13.0,
        "default": 4.5,
        "step": 0.1,
        "who_min": 0.5,
        "who_max": 4.0,
        "ideal": 2.0,
        "category": "Disinfection",
        "description": "Chlorine + ammonia disinfectant. Maximum safe: 4.0 ppm."
    },
    "Sulfate": {
        "name": "Sulfate",
        "unit": "mg/L",
        "min": 100.0,
        "max": 490.0,
        "default": 290.0,
        "step": 1.0,
        "who_min": 50.0,
        "who_max": 250.0,
        "ideal": 150.0,
        "category": "Minerals",
        "description": "Naturally occurring sulfur compounds. WHO limit: 250 mg/L."
    },
    "Conductivity": {
        "name": "Electrical Conductivity",
        "unit": "μS/cm",
        "min": 150.0,
        "max": 750.0,
        "default": 380.0,
        "step": 5.0,
        "who_min": 100.0,
        "who_max": 400.0,
        "ideal": 250.0,
        "category": "Physical",
        "description": "Ion concentration in water. WHO target: < 400 μS/cm."
    },
    "Organic_carbon": {
        "name": "Total Organic Carbon (TOC)",
        "unit": "ppm",
        "min": 2.0,
        "max": 28.0,
        "default": 11.5,
        "step": 0.1,
        "who_min": 1.0,
        "who_max": 4.0,
        "ideal": 2.0,
        "category": "Biological",
        "description": "Organic matter concentration. Purified drinking water < 4 ppm."
    },
    "Trihalomethanes": {
        "name": "Trihalomethanes (THMs)",
        "unit": "μg/L",
        "min": 0.5,
        "max": 125.0,
        "default": 55.0,
        "step": 0.5,
        "who_min": 5.0,
        "who_max": 80.0,
        "ideal": 20.0,
        "category": "Chemical",
        "description": "Carcinogenic disinfection by-products. Max safe limit: 80 μg/L."
    },
    "Turbidity": {
        "name": "Turbidity",
        "unit": "NTU",
        "min": 1.0,
        "max": 7.0,
        "default": 3.1,
        "step": 0.1,
        "who_min": 0.5,
        "who_max": 5.0,
        "ideal": 1.0,
        "category": "Physical",
        "description": "Cloudiness caused by suspended particles. WHO max: 5.0 NTU."
    }
}

PRESETS = {
    "mountain_spring": {
        "name": "Pristine Alpine Spring",
        "type": "Natural Fresh",
        "description": "Natural high-altitude glacial runoff with pristine clarity and balanced minerality.",
        "expected": "Potable",
        "values": {
            "ph": 7.4,
            "Hardness": 120.0,
            "Solids": 2400.0,
            "Chloramines": 1.2,
            "Sulfate": 185.0,
            "Conductivity": 280.0,
            "Organic_carbon": 3.1,
            "Trihalomethanes": 12.0,
            "Turbidity": 1.8
        }
    },
    "municipal_tap": {
        "name": "Municipal Treated Tap Water",
        "type": "Utility Drinking",
        "description": "Standard modern municipal tap water with controlled chlorination and filtration.",
        "expected": "Potable",
        "values": {
            "ph": 7.2,
            "Hardness": 160.0,
            "Solids": 12500.0,
            "Chloramines": 3.8,
            "Sulfate": 240.0,
            "Conductivity": 360.0,
            "Organic_carbon": 8.5,
            "Trihalomethanes": 45.0,
            "Turbidity": 2.5
        }
    },
    "industrial_polluted": {
        "name": "Industrial Runoff Contaminated",
        "type": "Hazardous Waste",
        "description": "Heavy industrial discharge with high dissolved solids, toxic chloramines, and sulfates.",
        "expected": "Non-Potable",
        "values": {
            "ph": 5.1,
            "Hardness": 285.0,
            "Solids": 48200.0,
            "Chloramines": 11.2,
            "Sulfate": 440.0,
            "Conductivity": 680.0,
            "Organic_carbon": 24.5,
            "Trihalomethanes": 112.0,
            "Turbidity": 6.2
        }
    },
    "acid_drainage": {
        "name": "Acid Mine Drainage",
        "type": "Highly Acidic Contaminant",
        "description": "Oxidized mining seepage with critical acidity and heavy sulfate minerals.",
        "expected": "Non-Potable",
        "values": {
            "ph": 3.4,
            "Hardness": 310.0,
            "Solids": 39500.0,
            "Chloramines": 6.5,
            "Sulfate": 475.0,
            "Conductivity": 720.0,
            "Organic_carbon": 17.8,
            "Trihalomethanes": 68.0,
            "Turbidity": 5.8
        }
    },
    "stagnant_lake": {
        "name": "Eutrophic Stagnant Pond",
        "type": "Biological Hazard",
        "description": "High organic matter decomposition, microbial growth, and intense turbidity.",
        "expected": "Non-Potable",
        "values": {
            "ph": 8.9,
            "Hardness": 215.0,
            "Solids": 29800.0,
            "Chloramines": 2.1,
            "Sulfate": 360.0,
            "Conductivity": 540.0,
            "Organic_carbon": 26.2,
            "Trihalomethanes": 88.0,
            "Turbidity": 6.6
        }
    },
    "overchlorinated": {
        "name": "Over-Chlorinated Pool Spill",
        "type": "Chemical Excess",
        "description": "Excessive disinfection dosing resulting in hazardous chloramines and trihalomethanes.",
        "expected": "Non-Potable",
        "values": {
            "ph": 6.2,
            "Hardness": 175.0,
            "Solids": 18000.0,
            "Chloramines": 12.8,
            "Sulfate": 310.0,
            "Conductivity": 490.0,
            "Organic_carbon": 12.0,
            "Trihalomethanes": 118.0,
            "Turbidity": 3.4
        }
    }
}

def analyze_water_quality(features_dict):
    """
    Evaluates WHO standard compliance and generates detailed parameter assessment.
    """
    compliance = {}
    violations = []
    cautions = []
    safe_count = 0

    for key, spec in FEATURE_CONFIG.items():
        val = float(features_dict.get(key, spec["default"]))
        who_min = spec["who_min"]
        who_max = spec["who_max"]
        
        status = "safe"
        severity = "none"
        message = f"Within standard ({who_min} - {who_max} {spec['unit']})"

        if val < who_min:
            status = "violation"
            severity = "critical"
            diff = round(who_min - val, 2)
            message = f"Below minimum standard by {diff} {spec['unit']} (Min: {who_min})"
            violations.append(f"{spec['name']} is too low ({val} {spec['unit']})")
        elif val > who_max:
            # Check if slight caution or critical violation
            if val > who_max * 1.25:
                status = "violation"
                severity = "critical"
                diff = round(val - who_max, 2)
                message = f"Exceeds maximum safe limit by {diff} {spec['unit']} (Max: {who_max})"
                violations.append(f"{spec['name']} exceeds safe threshold ({val} {spec['unit']})")
            else:
                status = "caution"
                severity = "moderate"
                diff = round(val - who_max, 2)
                message = f"Elevated above ideal limit by {diff} {spec['unit']} (Max: {who_max})"
                cautions.append(f"{spec['name']} is elevated ({val} {spec['unit']})")
        else:
            safe_count += 1

        compliance[key] = {
            "value": val,
            "status": status,
            "severity": severity,
            "message": message,
            "who_min": who_min,
            "who_max": who_max,
            "unit": spec["unit"],
            "name": spec["name"]
        }

    # Water Quality Index (0-100)
    wqi_score = int(round((safe_count / len(FEATURE_CONFIG)) * 100))

    # Treatment Recommendations
    recommendations = []
    ph_val = float(features_dict.get("ph", 7.0))
    if ph_val < 6.5:
        recommendations.append("Apply alkaline neutralization (Soda ash / Calcium carbonate injection) to raise pH to neutral range (7.0-7.5).")
    elif ph_val > 8.5:
        recommendations.append("Apply mild acid dosing (food-grade citric or sulfuric buffer) to neutralize alkaline excess.")
        
    solids_val = float(features_dict.get("Solids", 15000))
    if solids_val > 15000:
        recommendations.append("Install multi-stage Reverse Osmosis (RO) desalinization or membrane nanofiltration to reduce total dissolved solids.")

    chloramines_val = float(features_dict.get("Chloramines", 4.0))
    if chloramines_val > 4.0:
        recommendations.append("Implement Granular Activated Carbon (GAC) catalytic filtration or UV de-chlorination to break down excessive chloramines.")

    sulfate_val = float(features_dict.get("Sulfate", 250))
    if sulfate_val > 250:
        recommendations.append("Employ strong-base anion exchange softening resin or specialized RO membranes for sulfate ion extraction.")

    thms_val = float(features_dict.get("Trihalomethanes", 50))
    if thms_val > 80:
        recommendations.append("Aeration stripping tower and activated carbon adsorption required to eliminate toxic Trihalomethanes.")

    turbidity_val = float(features_dict.get("Turbidity", 3))
    if turbidity_val > 4.5:
        recommendations.append("Utilize coagulant flocculation (Alum/PAC) followed by dual-media rapid sand filtration to clarify suspended colloids.")

    if not recommendations:
        recommendations.append("Water meets standard physical and chemical benchmarks. Routine maintenance and periodic bacterial monitoring recommended.")

    return compliance, violations, cautions, wqi_score, recommendations

def calculate_3d_visual_params(features_dict, is_potable, potability_prob):
    """
    Computes real-time rendering parameters for the Three.js 3D water visualizer.
    """
    ph = float(features_dict.get("ph", 7.2))
    turbidity = float(features_dict.get("Turbidity", 3.0))
    solids = float(features_dict.get("Solids", 15000.0))
    chloramines = float(features_dict.get("Chloramines", 4.0))
    organic = float(features_dict.get("Organic_carbon", 12.0))
    thms = float(features_dict.get("Trihalomethanes", 50.0))
    
    # pH drives color hue:
    # Acidic (< 6.5): Amber/Reddish-Orange
    # Neutral (6.5 - 8.5): Pristine Azure Cyan / Aquamarine
    # Alkaline (> 8.5): Deep Indigo / Violet
    if ph < 6.0:
        # Acidic warm
        t = (6.0 - ph) / 6.0 # 0 to 1
        r = int(min(255, 30 + t * 220))
        g = int(max(40, 180 - t * 120))
        b = int(max(20, 240 - t * 200))
    elif ph > 8.5:
        # Alkaline violet
        t = min(1.0, (ph - 8.5) / 5.5)
        r = int(min(255, 30 + t * 160))
        g = int(max(30, 180 - t * 110))
        b = int(min(255, 230 + t * 25))
    else:
        # Neutral healthy water: Azure blue / crystal cyan
        r = 18
        g = 168
        b = 238

    # Non-potable tint alteration
    if not is_potable:
        # If contaminated, blend with murky olive or warning tint
        murk = min(1.0, (turbidity / 7.0) * 0.7 + (organic / 28.0) * 0.3)
        r = int(r * (1 - murk) + 110 * murk)
        g = int(g * (1 - murk) + 120 * murk)
        b = int(b * (1 - murk) + 60 * murk)

    water_color = f"#{r:02x}{g:02x}{b:02x}"
    
    # Turbidity maps to fog / liquid scattering (0.0 to 1.0)
    turbidity_factor = float(np.clip((turbidity - 1.0) / 6.0, 0.05, 0.95))
    
    # Particle count based on TDS and Turbidity
    particle_count = int(np.clip(30 + (solids / 50000.0) * 220 + (turbidity / 7.0) * 150, 30, 400))
    
    # Effervescent bubbles based on chloramines + THMs
    bubble_count = int(np.clip(15 + (chloramines / 13.0) * 120 + (thms / 125.0) * 80, 15, 250))
    bubble_speed = float(np.clip(0.5 + (chloramines / 10.0), 0.5, 2.8))
    
    # Wave agitation (fluid surface turbulence)
    wave_frequency = float(np.clip(1.0 + (turbidity / 3.0), 1.0, 3.5))
    wave_amplitude = float(np.clip(0.04 + (turbidity / 20.0), 0.03, 0.15))
    
    return {
        "water_color": water_color,
        "turbidity_factor": round(turbidity_factor, 3),
        "particle_count": particle_count,
        "bubble_count": bubble_count,
        "bubble_speed": round(bubble_speed, 2),
        "wave_frequency": round(wave_frequency, 2),
        "wave_amplitude": round(wave_amplitude, 3),
        "clarity_percent": round((1.0 - turbidity_factor) * 100, 1),
        "glow_intensity": round(1.2 if is_potable else 0.4, 2)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESETS)

@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    return jsonify({
        "metrics": model_metrics,
        "features": FEATURE_CONFIG
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    
    feature_keys = list(FEATURE_CONFIG.keys())
    features_dict = {}
    for feat in feature_keys:
        val = float(data.get(feat, FEATURE_CONFIG[feat]["default"]))
        features_dict[feat] = val
        
    X_input = pd.DataFrame([features_dict])[feature_keys]
    
    if model is not None:
        pred_class = int(model.predict(X_input)[0])
        pred_prob = float(model.predict_proba(X_input)[0][1])
    else:
        # Fallback heuristic if model file is reloading
        pred_class = 1 if 6.5 <= features_dict["ph"] <= 8.5 and features_dict["Solids"] < 25000 else 0
        pred_prob = 0.85 if pred_class == 1 else 0.15

    is_potable = bool(pred_class == 1)
    potable_probability = round(pred_prob * 100, 1)
    non_potable_probability = round((1.0 - pred_prob) * 100, 1)

    compliance, violations, cautions, wqi_score, recommendations = analyze_water_quality(features_dict)
    visual_params = calculate_3d_visual_params(features_dict, is_potable, pred_prob)

    # Risk level classification
    if is_potable and potable_probability >= 75:
        risk_level = "Safe / Optimal"
        risk_badge = "success"
    elif is_potable:
        risk_level = "Acceptable / Marginally Potable"
        risk_badge = "warning"
    elif potable_probability >= 30:
        risk_level = "Unsafe / High Contaminant Risk"
        risk_badge = "danger"
    else:
        risk_level = "Critical Hazard / Toxic"
        risk_badge = "critical"

    return jsonify({
        "is_potable": is_potable,
        "potability_class": pred_class,
        "potability_probability": potable_probability,
        "non_potable_probability": non_potable_probability,
        "verdict": "SAFE TO DRINK" if is_potable else "UNSAFE - NON-POTABLE",
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "wqi_score": wqi_score,
        "compliance": compliance,
        "violations": violations,
        "cautions": cautions,
        "recommendations": recommendations,
        "visual_params": visual_params
    })

@app.route("/api/what-if", methods=["POST"])
def what_if_optimization():
    """
    Computes the minimal necessary adjustments to turn an unsafe sample into potable.
    """
    data = request.get_json() or {}
    adjusted = {}
    changes = []
    
    for key, spec in FEATURE_CONFIG.items():
        val = float(data.get(key, spec["default"]))
        who_min = spec["who_min"]
        who_max = spec["who_max"]
        ideal = spec["ideal"]
        
        if val < who_min:
            target = who_min + (who_max - who_min) * 0.3
            adjusted[key] = round(target, 2)
            changes.append({
                "feature": spec["name"],
                "unit": spec["unit"],
                "current": val,
                "suggested": round(target, 2),
                "action": f"Increase to safe range ({who_min} - {who_max})"
            })
        elif val > who_max:
            target = who_min + (who_max - who_min) * 0.7
            adjusted[key] = round(target, 2)
            changes.append({
                "feature": spec["name"],
                "unit": spec["unit"],
                "current": val,
                "suggested": round(target, 2),
                "action": f"Reduce to compliant threshold (< {who_max})"
            })
        else:
            adjusted[key] = val

    # Verify prediction with adjusted parameters
    feat_keys = list(FEATURE_CONFIG.keys())
    X_adj = pd.DataFrame([adjusted])[feat_keys]
    if model is not None:
        adj_prob = float(model.predict_proba(X_adj)[0][1])
        adj_potable = bool(model.predict(X_adj)[0] == 1)
    else:
        adj_prob = 0.88
        adj_potable = True

    return jsonify({
        "adjusted_parameters": adjusted,
        "changes": changes,
        "projected_potability_probability": round(adj_prob * 100, 1),
        "projected_potable": adj_potable
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
