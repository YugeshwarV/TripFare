# TripFare: NYC Taxi Fare Prediction App

A Streamlit-powered machine learning web app that predicts the total taxi fare in NYC based on trip details like distance, time, tip, tolls, and more.

---

## Problem Statement

As part of an urban mobility analytics project, the goal is to predict the **total fare amount** for NYC taxi trips using historical ride data. This helps riders estimate costs, supports driver incentives, and enables pricing transparency.

---

## Project Highlights

- Predicts **`total_amount`** (target)
- Includes surcharges like `tip_amount`, `mta_tax`, `tolls_amount`, `improvement_surcharge`
- Regression modeling using **Gradient Boosting**
- Clean Streamlit UI for real-time prediction

---

## Dataset Description

The model was trained on real NYC Yellow Taxi data with the following features:

| Feature                 | Description                                 |
|------------------------|---------------------------------------------|
| `trip_distance_km`     | Distance between pickup and dropoff (in km) |
| `trip_duration_min`    | Duration of the trip (in minutes)           |
| `fare_per_km`          | Fare per km                                 |
| `fare_per_min`         | Fare per minute                             |
| `hour`                 | Hour of the pickup                          |
| `RatecodeID`           | Fare type code (standard, JFK, etc.)        |
| `passenger_count`      | Number of passengers                        |
| `VendorID`             | Vendor/Taxi operator ID                     |

---

## Model Overview

- Algorithm: **GradientBoostingRegressor**
- Trained on: 80% of cleaned dataset
- Tuned using: `RandomizedSearchCV`
- Final model accuracy: **R² ~ 0.895**

---

## Files Included

| File Name             | Purpose                                 |
|-----------------------|------------------------------------------|
| `TripFareApp.py`      | Streamlit app for predicting fare        |
| `GB_model.pkl`        | Trained Gradient Boosting model          |
| `scaler.pkl`          | StandardScaler used in training          |
| `feature_columns.pkl` | Order of features expected by model      |

---

## How to Run the App

### 1. Clone this repo or download the files
```bash
git clone https://github.com/your-username/tripfare-predictor.git
cd tripfare-predictor
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit scikit-learn numpy pandas xgboost
```

### 3. Run the app
```bash
streamlit run TripFareApp.py
```

Then open `http://localhost:8501` in your browser.

---

## How Fare is Calculated

```
Estimated Total Fare = 
    predicted_fare_amount (from ML model) +
    tip_amount (user input) +
    tolls_amount (user input) +
    mta_tax (fixed at $0.50) +
    improvement_surcharge (fixed at $0.30)
```

---

## Model Training Overview

- Outlier Removal using IQR for distance, fare, duration
- Feature Engineering: fare_per_km, fare_per_min, hour
- Feature Scaling: StandardScaler
- Model Evaluation: R², RMSE, MAE
- Final Model: Tuned GradientBoostingRegressor

---

## Sample Prediction

> **Input:** 5 km, 15 min, tip = $2, tolls = $0  
> **Prediction:** Estimated Fare = `$13.45`

---

## Acknowledgements

- NYC Taxi and Limousine Commission (TLC) for the dataset
- Scikit-learn for modeling tools
- Streamlit for interactive deployment

---

## Author

Built by **Yugeshwar V**  

