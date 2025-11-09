import streamlit as st
import requests

# ---------------- CONFIG ----------------
API_URL = "https://eta-predictor.onrender.com/predict"

st.set_page_config(page_title="ETA Predictor", page_icon="🚚", layout="centered")

st.title("🚚 ETA Prediction App")
st.write("Predict delivery or pickup ETA using the trained CatBoost model deployed on Flask + Docker.")

# ---------------- INPUT FORM ----------------
mode = st.radio("Select Mode", ["pickup", "delivery"])

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", ["Chongqing", "Shanghai", "Hangzhou", "Yantai"])
    lng = st.number_input("Longitude", value=106.55 if city == "Chongqing" else 121.47)
    lat = st.number_input("Latitude", value=29.56 if city == "Chongqing" else 31.23)
    aoi_type = st.selectbox("AOI Type (0/1)", [0, 1])
    distance = st.number_input(
        f"{'Pickup' if mode == 'pickup' else 'Delivery'} Distance (km)", value=2.5, step=0.1
    )

with col2:
    accept_hour = st.number_input("Accept Hour (0–23)", value=10)
    target_hour = st.number_input(f"{'Pickup' if mode == 'pickup' else 'Delivery'} Hour (0–23)", value=11)
    accept_day = st.number_input("Accept Day (1–31)", value=9)
    target_day = st.number_input(f"{'Pickup' if mode == 'pickup' else 'Delivery'} Day (1–31)", value=9)
    accept_month = st.number_input("Accept Month (1–12)", value=10)
    target_month = st.number_input(f"{'Pickup' if mode == 'pickup' else 'Delivery'} Month (1–12)", value=10)

hour_bucket = st.selectbox("Hour Bucket", ["Morning", "Afternoon", "Evening", "Night"])
day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict ETA"):
    st.info("Sending data to Flask API for prediction...")

    payload = {
        "mode": mode,
        "features": {
            "city": city,
            "lng": lng,
            "lat": lat,
            "aoi_type": aoi_type,
            f"{mode}_distance_km": distance,
            "accept_hour": accept_hour,
            f"{mode}_hour": target_hour,
            "accept_day": accept_day,
            f"{mode}_day": target_day,
            "accept_month": accept_month,
            f"{mode}_month": target_month,
            "accept_date": "2025-10-09",
            f"{mode}_date": "2025-10-09",
            "hour_bucket": hour_bucket,
            "day_type": day_type,
        },
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ ETA Predicted: {data['eta_minutes']} minutes")
            st.write(f"Normalized ETA: `{data['eta_normalized']}`")
            st.write(f"Processing Time: `{data['processing_time_sec']} sec`")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

st.markdown("---")
st.caption("Backend: Flask + Docker | Frontend: Streamlit | Model: CatBoost")
