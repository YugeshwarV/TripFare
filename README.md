
# 🚖 NYC Taxi Fare Prediction App

A machine learning web app that predicts the total fare for NYC taxi rides based on trip details such as pickup/dropoff locations, timestamps, and passenger count. Built using **Streamlit**, **scikit-learn**, and a trained **Linear Regression model**.

---

## 📦 Features

- Predicts **total fare** using a regression model trained on NYC taxi data.
- User-friendly **Streamlit interface**.
- Accepts detailed trip inputs like:
  - Pickup & Dropoff datetime
  - Passenger count
  - Coordinates
  - Extra charges, tax, tips, etc.
- Final prediction includes **base fare + surcharges**.

---

## 🧠 Model Overview

- Model: `LinearRegression`
- Features used:
  - Numeric: `VendorID`, `passenger_count`, lat/lon coordinates, trip duration, hour, weekday
  - Engineered: `trip_duration_min`, `trip_distance_km`
  - One-hot encoded: `RatecodeID_*`, `payment_type_*`
- Target variable: `fare_amount`
- Preprocessing: `StandardScaler` for normalization

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nyc-taxi-fare-predictor.git
cd nyc-taxi-fare-predictor
```

### ✅ 2. Install Dependencies

It's recommended to use a virtual environment:

```bash
conda create -n taxi-fare python=3.10
conda activate taxi-fare
pip install -r requirements.txt
```

Required packages:
```txt
streamlit
pandas
numpy
scikit-learn==1.5.1
```

### ✅ 3. Place Artifacts

Ensure the following trained model files are in the project root:

- `model.pkl`
- `scaler.pkl`
- `expected_columns.pkl`

> If not present, you can regenerate using the training notebook or script provided.

---

## 🚀 Running the App

```bash
streamlit run Tripfare_app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📂 File Structure

```
├── cleaned_taxi_data.csv        # Preprocessed dataset
├── TripFareNB.ipynb             # Jupyter notebook used to train the model
├── Tripfare_app.py              # Streamlit app
├── model.pkl                    # Trained LinearRegression model
├── scaler.pkl                   # Fitted StandardScaler
├── expected_columns.pkl         # List of model's expected input columns
├── README.md                    # Project documentation
```

---

## 💡 Prediction Logic

1. User inputs trip details via Streamlit.
2. Inputs are processed and one-hot encoded.
3. Scaled using `StandardScaler`.
4. Model predicts **base fare**.
5. Final fare is calculated as:

```python
final_fare = base_fare + extra + mta_tax + tip + tolls + improvement_surcharge
```

---

## 📊 Example Input

| Field            | Example         |
|------------------|-----------------|
| Pickup Date      | 2025-06-14      |
| Pickup Time      | 09:00 AM        |
| Dropoff Time     | 09:30 AM        |
| VendorID         | 2               |
| Passenger Count  | 1               |
| Pickup Lat/Lon   | 40.748817 / -73.985428 |
| Dropoff Lat/Lon  | 40.751652 / -73.977295 |
| Ratecode ID      | 1               |
| Payment Type     | 2               |
| Extra            | 0.5             |
| MTA Tax          | 0.5             |
| Tip              | 2.0             |
| Tolls            | 0.0             |
| Surcharge        | 0.3             |
---

## 👨‍💻 Author

**Yugeshwar**  

