import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, time
from math import radians, sin, cos, sqrt, asin

# ----------------------------
# Load Pickled Artifacts
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("expected_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="NYC Taxi Fare Estimator", layout="centered")
st.title("🚖 NYC Taxi Fare Estimator")
st.markdown("Enter trip details below to predict the total fare amount.")

# ----------------------------
# Input Fields
# ----------------------------
pickup_date = st.date_input("Pickup Date", datetime.today())
pickup_time = st.time_input("Pickup Time", time(9, 0))
dropoff_date = st.date_input("Dropoff Date", datetime.today())
dropoff_time = st.time_input("Dropoff Time", time(9, 30))

pickup_datetime = datetime.combine(pickup_date, pickup_time)
dropoff_datetime = datetime.combine(dropoff_date, dropoff_time)

if dropoff_datetime <= pickup_datetime:
    st.warning("⚠️ Dropoff time must be after pickup time.")
    st.stop()

pickup_lat = st.number_input("Pickup Latitude", value=40.748817)
pickup_long = st.number_input("Pickup Longitude", value=-73.985428)
dropoff_lat = st.number_input("Dropoff Latitude", value=40.748817)
dropoff_long = st.number_input("Dropoff Longitude", value=-73.985428)

passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
vendor_id = st.selectbox("Vendor ID", [1, 2])
ratecode_id = st.selectbox("Ratecode ID", [1, 2, 3, 4, 5, 6])
store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["N", "Y"])
payment_type = st.selectbox("Payment Type", [1, 2, 3, 4, 5, 6])
fare_amount = st.number_input("Fare Amount (Base)", value=12.5)
tip_amount = st.number_input("Tip Amount", value=2.0)

# ----------------------------
# Feature Engineering
# ----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0)**2
    c = 2 * asin(sqrt(a))
    return R * c

trip_distance_km = haversine_distance(pickup_lat, pickup_long, dropoff_lat, dropoff_long)
trip_duration_min = (dropoff_datetime - pickup_datetime).total_seconds() / 60
hour = pickup_datetime.hour
day_of_week = pickup_datetime.weekday()
store_and_fwd_flag = 1 if store_and_fwd_flag == "Y" else 0

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Fare"):
    input_data = {
        'trip_distance_km': round(trip_distance_km, 3),
        'trip_duration_min': round(trip_duration_min, 2),
        'tip_amount': tip_amount,
        'fare_amount': fare_amount,
        'passenger_count': passenger_count,
        'hour': hour,
        'day_of_week': day_of_week,
        'RatecodeID': ratecode_id,
        'payment_type': payment_type,
        'VendorID': vendor_id,
        'store_and_fwd_flag': store_and_fwd_flag
    }

    input_df = pd.DataFrame([input_data])

    # Reindex to match training features
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Optional Debug View
    # st.write("🧪 Input data passed to model:")
    # st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)[0]
        prediction = max(prediction, 1.00)  # Avoid negative or zero prediction
        st.success(f"💰 Estimated Total Fare: ${prediction:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
