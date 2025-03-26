# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:51:54 2025

@author: eulal
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ===== 1. Load Test Data (2019) =====
test_file = r"C:\Users\eulal\Documents\project 1 python\testData_2019_Civil - testData_2019_Civil.csv"
test_data = pd.read_csv(test_file)

# Rename necessary columns
test_data.rename(columns={"Civil (kWh)": "Power_kW", "Date": "Timestamp"}, inplace=True)
test_data["Timestamp"] = pd.to_datetime(test_data["Timestamp"], errors='coerce')

# ===== 2. Create Features (as in training) =====
test_data["Power-1"] = test_data["Power_kW"].shift(1)
test_data["Hour"] = test_data["Timestamp"].dt.hour
test_data["Weekday"] = test_data["Timestamp"].dt.weekday

# Add holiday flag as Day Type
holidays = ["2019-01-01", "2019-04-25", "2019-05-01", "2019-06-10", "2019-12-25"]
test_data["Day Type"] = test_data["Weekday"]
test_data.loc[test_data["Timestamp"].dt.strftime("%Y-%m-%d").isin(holidays), "Day Type"] = 7

# Remove missing values (from shift)
test_data.dropna(inplace=True)

# ===== 3. Load model and scaler =====
model_path = r"C:\Users\eulal\Documents\project 1 python\modelo_previsao.pkl"
scaler_path = r"C:\Users\eulal\Documents\project 1 python\scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ===== 4. Prepare features and make predictions =====
features = ["Power-1", "Day Type", "Hour", "temp_C"]
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

# ===== 5. Create forecast DataFrame =====
forecast_df = pd.DataFrame({
    "Timestamp": test_data["Timestamp"],
    "Power_kW": test_data["Power_kW"],
    "Predicted_Power_kW": y_pred
})

# ===== 6. Filter to Jan–Mar 2019 only =====
forecast_df = forecast_df[
    (forecast_df["Timestamp"] >= "2019-01-01") &
    (forecast_df["Timestamp"] < "2019-04-01")
]

# ===== 7. Save to CSV =====
output_path = r"C:\Users\eulal\Documents\project 1 python\forecast_jan_mar_2019.csv"
forecast_df.to_csv(output_path, index=False)

print("✅ Forecast file generated and saved as forecast_jan_mar_2019.csv")
