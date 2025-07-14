import streamlit as st
import numpy as np
import pickle

# Load the model, scaler, and feature columns
with open("GB_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

st.set_page_config(page_title="TripFare Estimator", layout="centered")
st.title("🚕 Trip Fare Estimator")

# Input form
st.subheader("Enter Trip Details")

trip_distance_km = st.number_input("Trip Distance (km)", min_value=0.0, value=5.0, step=0.1)
trip_duration_min = st.number_input("Trip Duration (minutes)", min_value=0.0, value=10.0, step=1.0)
fare_per_km = st.number_input("Fare per km", min_value=0.0, value=2.5, step=0.1)
fare_per_min = st.number_input("Fare per minute", min_value=0.0, value=0.5, step=0.1)
hour = st.slider("Pickup Hour (0-23)", min_value=0, max_value=23, value=10)
RatecodeID = st.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 6])
passenger_count = st.slider("Passenger Count", min_value=1, max_value=6, value=1)
VendorID = st.selectbox("Vendor ID", [1, 2])
tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, value=0.0, step=0.1)
tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, value=0.0, step=0.1)

if st.button("Estimate Fare"):
    # Fixed surcharges
    mta_tax = 0.50
    improvement_surcharge = 0.30

    # Prepare input data
    input_data = np.array([[
        trip_distance_km, trip_duration_min, fare_per_km, fare_per_min,
        hour, RatecodeID, passenger_count, VendorID
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict base fare (fare_amount)
    base_fare = model.predict(input_scaled)[0]

    # Calculate total fare
    total_fare = base_fare + improvement_surcharge + mta_tax + tip_amount + tolls_amount

    st.success(f"💵 Estimated Total Fare: **${total_fare:.2f}**")
