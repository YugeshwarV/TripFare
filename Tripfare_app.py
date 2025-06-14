import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, time

# ------------------- Load Saved Artifacts -------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("expected_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="centered")
st.title("🚖 NYC Taxi Fare Prediction")

st.markdown("Enter trip details below to estimate the total fare:")

# Date and time inputs
pickup_date = st.date_input("Pickup Date", datetime.today())
pickup_time = st.time_input("Pickup Time", time(9, 0))
pickup_datetime = datetime.combine(pickup_date, pickup_time)

dropoff_date = st.date_input("Dropoff Date", datetime.today())
dropoff_time = st.time_input("Dropoff Time", time(9, 30))
dropoff_datetime = datetime.combine(dropoff_date, dropoff_time)

# Prevent invalid datetime
if dropoff_datetime <= pickup_datetime:
    st.warning("⚠️ Dropoff time must be after pickup time.")
else:
    # Trip detail inputs
    vendor_id = st.selectbox("Vendor ID", [1, 2])
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
    pickup_long = st.number_input("Pickup Longitude", value=-73.985428)
    pickup_lat = st.number_input("Pickup Latitude", value=40.748817)
    dropoff_long = st.number_input("Dropoff Longitude", value=-73.985428)
    dropoff_lat = st.number_input("Dropoff Latitude", value=40.748817)
    ratecode_id = st.selectbox("Ratecode ID", [1, 2, 3, 4, 5, 6])
    store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["N", "Y"])
    payment_type = st.selectbox("Payment Type", [1, 2, 3, 4, 5, 6])
    extra = st.number_input("Extra Charges", value=0.5)
    mta_tax = st.number_input("MTA Tax", value=0.5)
    tip_amount = st.number_input("Tip Amount", value=0.0)
    tolls_amount = st.number_input("Tolls Amount", value=0.0)
    improvement_surcharge = st.number_input("Improvement Surcharge", value=0.3)

    # Compute trip duration
    duration_minutes = (dropoff_datetime - pickup_datetime).total_seconds() / 60

    # ------------------- Prediction -------------------
    if st.button("Predict Fare"):
        # Base features
        input_dict = {
            "VendorID": vendor_id,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_long,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_long,
            "dropoff_latitude": dropoff_lat,
            "store_and_fwd_flag": 1 if store_and_fwd_flag == "Y" else 0,
            "extra": extra,
            "mta_tax": mta_tax,
            "tip_amount": tip_amount,
            "tolls_amount": tolls_amount,
            "improvement_surcharge": improvement_surcharge,
            "duration_minutes": duration_minutes,
            "hour": pickup_datetime.hour,
            "day_of_week": pickup_datetime.weekday(),
            "RatecodeID": ratecode_id,
            "payment_type": payment_type
        }

        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, columns=["RatecodeID", "payment_type"], prefix=["RatecodeID", "payment_type"])

        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns

        # Reorder columns to match training data
        input_df = input_df[expected_columns]

        # Scale numeric features
        scaled_input = scaler.transform(input_df)

        # Predict base fare
        base_fare = model.predict(scaled_input)[0]
        base_fare = max(base_fare, 1.00)
        
        # Add all surcharges to the base fare
        final_fare = base_fare + extra + mta_tax + tip_amount + tolls_amount + improvement_surcharge
        final_fare = max(final_fare, 1.00)  # Ensure minimum fare
        
        st.success(f"💰 Estimated Total Fare: ${final_fare:.2f}")
