import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("cleaned_taxi_data.csv")

# Drop unwanted or non-numeric columns
exclude_columns = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "fare_amount",
    "total_amount", "fare_per_km", "fare_per_min",
    "log_total_amount", "log_distance", "log_duration"
]
X = df.drop(columns=exclude_columns)
y = df["fare_amount"]

# Keep only numeric columns
X = X.select_dtypes(include=["number"])

# Save expected column names
expected_columns = list(X.columns)

# Fit scaler and model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Save artifacts
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("expected_columns.pkl", "wb") as f:
    pickle.dump(expected_columns, f)
