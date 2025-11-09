import json
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
import logging
import os

# -------------------- CONFIGURATION --------------------
app = Flask(__name__)

# Setup Logging
logging.basicConfig(
    filename="eta_api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Load models
pickup_model = CatBoostRegressor()
pickup_model.load_model("catboost_pickup_eta_model.cbm")

delivery_model = CatBoostRegressor()
delivery_model.load_model("catboost_delivery_eta_model.cbm")

print("✅ Models loaded successfully (Pickup & Delivery)")

# Load scaling parameters
with open("pickup_scaling_params.json", "r") as f:
    pickup_scaling = json.load(f)
with open("delivery_scaling_params.json", "r") as f:
    delivery_scaling = json.load(f)
with open("eta_scaling_params.json", "r") as f:
    eta_scaling = json.load(f)


# -------------------- HELPER FUNCTIONS --------------------
def normalize_value(val, col, mode):
    """Normalize real-world input using saved scaling parameters."""
    scaling_dict = pickup_scaling if mode == "pickup" else delivery_scaling
    if col not in scaling_dict:
        return val
    vmin = scaling_dict[col]["min"]
    vmax = scaling_dict[col]["max"]
    if vmax == vmin:
        return 0
    return (val - vmin) / (vmax - vmin)


def preprocess_payload(payload, mode):
    """Convert input payload to properly formatted DataFrame for prediction."""
    df = pd.DataFrame([payload])

    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            df[c] = df[c].astype(str)

    cat_features = (
        ['city', 'accept_date', 'pickup_date', 'hour_bucket', 'day_type']
        if mode == "pickup"
        else ['city', 'accept_date', 'delivery_date', 'day_type', 'hour_bucket']
    )

    for cat in cat_features:
        if cat in df.columns:
            df[cat] = df[cat].astype(str)

    expected_cols = (
        ['city', 'lng', 'lat', 'aoi_type', 'pickup_distance_km',
         'accept_hour', 'pickup_hour', 'accept_day', 'pickup_day',
         'accept_month', 'pickup_month', 'accept_date', 'pickup_date',
         'hour_bucket', 'day_type']
        if mode == "pickup"
        else ['city', 'lng', 'lat', 'aoi_type', 'delivery_distance_km',
              'accept_hour', 'delivery_hour', 'accept_day', 'delivery_day',
              'accept_month', 'delivery_month', 'accept_date', 'delivery_date',
              'day_type', 'hour_bucket']
    )

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[expected_cols]


def predict_eta(payload, mode):
    """Predict ETA (actual minutes) from real input data."""
    for col, val in payload.items():
        if isinstance(val, (int, float)):
            payload[col] = normalize_value(val, col, mode)

    X = preprocess_payload(payload, mode)
    model = pickup_model if mode == "pickup" else delivery_model

    eta_norm = float(model.predict(X)[0])

    eta_min = eta_scaling[mode]["eta_min"]
    eta_max = eta_scaling[mode]["eta_max"]
    eta_actual = eta_norm * (eta_max - eta_min) + eta_min

    return eta_actual, eta_norm


# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return jsonify({
        "message": "✅ ETA Prediction API is running.",
        "usage": "Send a POST request to /predict with JSON data (mode + features)."
    })


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        data = request.get_json(force=True)

        # allow both formats (with or without "features")
        mode = data.get("mode", "pickup")
        payload = data.get("features", data)

        if mode not in ["pickup", "delivery"]:
            return jsonify({"error": "Invalid mode. Must be 'pickup' or 'delivery'"}), 400

        eta_actual, eta_norm = predict_eta(payload, mode)

        response = {
            "mode": mode,
            "eta_normalized": round(eta_norm, 4),
            "eta_minutes": round(eta_actual, 2),
            "processing_time_sec": round(time.time() - start_time, 3)
        }

        # Logging for monitoring
        logging.info(f"Mode: {mode} | Payload: {payload} | ETA: {response}")

        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# -------------------- MANUAL TEST MODE --------------------
def manual_test():
    mode = input("Enter mode (pickup/delivery): ").strip().lower()
    if mode not in ["pickup", "delivery"]:
        print("❌ Invalid mode. Please enter 'pickup' or 'delivery'.")
        return

    print(f"\n🚀 Running manual ETA test for {mode.upper()} mode...\n")

    if mode == "pickup":
        payload = {
            "city": "Chongqing",
            "lng": 106.55,
            "lat": 29.56,
            "aoi_type": 1,
            "pickup_distance_km": 2.3,
            "accept_hour": 10,
            "pickup_hour": 11,
            "accept_day": 9,
            "pickup_day": 9,
            "accept_month": 10,
            "pickup_month": 10,
            "accept_date": "2025-10-09",
            "pickup_date": "2025-10-09",
            "hour_bucket": "Afternoon",
            "day_type": "Weekday"
        }
    else:
        payload = {
            "city": "Chongqing",
            "lng": 106.55,
            "lat": 29.56,
            "aoi_type": 1,
            "delivery_distance_km": 2.8,
            "accept_hour": 10,
            "delivery_hour": 11,
            "accept_day": 9,
            "delivery_day": 9,
            "accept_month": 10,
            "delivery_month": 10,
            "accept_date": "2025-10-09",
            "delivery_date": "2025-10-09",
            "day_type": "Weekday",
            "hour_bucket": "Afternoon"
        }

    eta_actual, eta_norm = predict_eta(payload, mode)
    eta_hours = int(eta_actual // 60)
    eta_minutes = int(eta_actual % 60)
    print(f"🔹 Normalized ETA prediction: {eta_norm:.4f}")
    print(f"⏱️ Actual ETA: {eta_actual:.2f} minutes (~{eta_hours}h {eta_minutes}m)")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    in_docker = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"

    if in_docker:
        print("🚀 Running inside Docker — starting Flask API automatically...")
        app.run(host="0.0.0.0", port=5000)
    else:
        mode = input("Run API server or manual test? (api/manual): ").strip().lower()
        if mode == "api":
            app.run(host="0.0.0.0", port=5000)
        else:
            manual_test()
