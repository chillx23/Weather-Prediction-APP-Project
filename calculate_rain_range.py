import os
import pickle
import numpy as np
import pandas as pd

# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ============================================================
# FILE PATHS
# ============================================================

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "Rain_Models",
    "rain_xgboost_test_predictions.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Rain_Models",
    "rain_prediction_range.pkl"
)

# ============================================================
# CHECK FILE
# ============================================================

print("Loading rain predictions...")

if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(
        f"\nRain prediction file not found:\n"
        f"{PREDICTIONS_PATH}\n\n"
        f"Make sure you have run:\n"
        f"Models/rain_xgboost.py"
    )

# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTIONS_PATH
)

print("Rain predictions loaded!")

print("Columns:")
print(df.columns.tolist())

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "actual_rain",
    "xgboost_prediction"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' "
            f"not found in prediction file."
        )

# ============================================================
# GET VALUES
# ============================================================

actual = df[
    "actual_rain"
].values

prediction = df[
    "xgboost_prediction"
].values

# ============================================================
# CALCULATE ERRORS
# ============================================================

errors = actual - prediction

# ============================================================
# ERROR METRICS
# ============================================================

mae = np.mean(
    np.abs(errors)
)

rmse = np.sqrt(
    np.mean(errors ** 2)
)

# ============================================================
# TIGHTER RANGE
# ============================================================

# Middle 50% of historical errors

lower_error = np.percentile(
    errors,
    25
)

upper_error = np.percentile(
    errors,
    75
)

# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 50)
print("RAIN ERROR ANALYSIS")
print("=" * 50)

print(
    f"MAE              : {mae:.4f}"
)

print(
    f"RMSE             : {rmse:.4f}"
)

print()
print("TIGHTER ERROR RANGE")
print("-------------------")

print(
    f"25% error        : {lower_error:.4f}"
)

print(
    f"75% error        : {upper_error:.4f}"
)

# ============================================================
# SAVE RANGE INFORMATION
# ============================================================

range_data = {

    "mae": float(mae),

    "rmse": float(rmse),

    "lower_error": float(
        lower_error
    ),

    "upper_error": float(
        upper_error
    ),

    "percentile_lower": 25,

    "percentile_upper": 75
}

with open(
    OUTPUT_PATH,
    "wb"
) as file:

    pickle.dump(
        range_data,
        file
    )

# ============================================================
# VERIFY FILE
# ============================================================

print()
print("=" * 50)
print("FILES SAVED")
print("=" * 50)

print(
    OUTPUT_PATH
)

print()
print(
    "Rain prediction range information created!"
)

print()
print(
    "The rain range now uses the middle 50%"
)

print(
    "of historical prediction errors."
)