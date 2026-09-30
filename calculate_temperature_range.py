import os
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "Temperature_Models",
    "temperature_weighted_ensemble_predictions.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Temperature_Models",
    "temperature_prediction_range.pkl"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("Loading ensemble predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

print("Columns:")
print(df.columns.tolist())


# ============================================================
# FIND COLUMNS
# ============================================================

actual_column = "actual_temperature"
prediction_column = "ensemble_prediction"


if actual_column not in df.columns:
    raise ValueError(
        f"'{actual_column}' not found in prediction file."
    )

if prediction_column not in df.columns:
    raise ValueError(
        f"'{prediction_column}' not found in prediction file."
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df[
    df[actual_column].notna()
    & df[prediction_column].notna()
].copy()


# ============================================================
# CALCULATE ERRORS
# ============================================================

df["error"] = (
    df[actual_column]
    - df[prediction_column]
)

df["absolute_error"] = abs(df["error"])


# ============================================================
# CALCULATE ERROR STATISTICS
# ============================================================

mae = df["absolute_error"].mean()

rmse = np.sqrt(
    np.mean(df["error"] ** 2)
)

lower_error = np.percentile(
    df["error"],
    2.5
)

upper_error = np.percentile(
    df["error"],
    97.5
)


# ============================================================
# SAVE RANGE INFORMATION
# ============================================================

range_data = {
    "mae": float(mae),
    "rmse": float(rmse),
    "lower_error": float(lower_error),
    "upper_error": float(upper_error)
}


with open(OUTPUT_PATH, "wb") as file:
    pickle.dump(range_data, file)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 50)
print("TEMPERATURE ERROR ANALYSIS")
print("=" * 50)

print(f"MAE              : {mae:.4f} °C")
print(f"RMSE             : {rmse:.4f} °C")
print(f"2.5% error       : {lower_error:.4f} °C")
print(f"97.5% error      : {upper_error:.4f} °C")

print()
print("=" * 50)
print("FILES SAVED")
print("=" * 50)

print(OUTPUT_PATH)

print()
print("Temperature prediction range information created!")