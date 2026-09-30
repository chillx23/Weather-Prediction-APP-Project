import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

MODEL_FOLDER = "Temperature_Models"

RIDGE_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_ridge_test_predictions.csv"
)

XGB_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_xgboost_test_predictions.csv"
)

LSTM_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_test_predictions.csv"
)


# ============================================================
# 2. LOAD PREDICTION FILES
# ============================================================

print("Loading model predictions...")

ridge_df = pd.read_csv(RIDGE_FILE)
xgb_df = pd.read_csv(XGB_FILE)
lstm_df = pd.read_csv(LSTM_FILE)


print("\nFiles loaded successfully!")

print("Ridge:", ridge_df.shape)
print("XGBoost:", xgb_df.shape)
print("LSTM:", lstm_df.shape)


# ============================================================
# 3. CONVERT DATETIME
# ============================================================

ridge_df["datetime"] = pd.to_datetime(
    ridge_df["datetime"]
)

xgb_df["datetime"] = pd.to_datetime(
    xgb_df["datetime"]
)

lstm_df["datetime"] = pd.to_datetime(
    lstm_df["datetime"]
)


# ============================================================
# 4. RENAME COLUMNS
# ============================================================

ridge_df = ridge_df.rename(
    columns={
        "temperature_target_next_day":
            "actual_temperature",

        "ridge_prediction":
            "ridge_prediction"
    }
)

xgb_df = xgb_df.rename(
    columns={
        "temperature_target_next_day":
            "actual_temperature_xgb",

        "xgboost_prediction":
            "xgboost_prediction"
    }
)

lstm_df = lstm_df.rename(
    columns={
        "temperature_target_next_day":
            "actual_temperature_lstm",

        "lstm_prediction":
            "lstm_prediction"
    }
)


# ============================================================
# 5. SELECT REQUIRED COLUMNS
# ============================================================

ridge_df = ridge_df[
    [
        "datetime",
        "city",
        "actual_temperature",
        "ridge_prediction"
    ]
]

xgb_df = xgb_df[
    [
        "datetime",
        "city",
        "actual_temperature_xgb",
        "xgboost_prediction"
    ]
]

lstm_df = lstm_df[
    [
        "datetime",
        "city",
        "actual_temperature_lstm",
        "lstm_prediction"
    ]
]


# ============================================================
# 6. MERGE RIDGE + XGBOOST
# ============================================================

ensemble_df = pd.merge(
    ridge_df,
    xgb_df,
    on=["datetime", "city"],
    how="inner"
)


# ============================================================
# 7. MERGE LSTM
# ============================================================

ensemble_df = pd.merge(
    ensemble_df,
    lstm_df,
    on=["datetime", "city"],
    how="inner"
)


print("\n==========================================")
print("MERGE RESULTS")
print("==========================================")

print(
    "Rows after merging:",
    len(ensemble_df)
)


# ============================================================
# 8. CHECK ACTUAL VALUES
# ============================================================

print("\nChecking actual temperature values...")

ridge_actual = ensemble_df[
    "actual_temperature"
]

xgb_actual = ensemble_df[
    "actual_temperature_xgb"
]

lstm_actual = ensemble_df[
    "actual_temperature_lstm"
]


ridge_xgb_match = np.allclose(
    ridge_actual,
    xgb_actual,
    equal_nan=True
)

ridge_lstm_match = np.allclose(
    ridge_actual,
    lstm_actual,
    equal_nan=True
)


if not ridge_xgb_match:

    print(
        "WARNING: Ridge and XGBoost actual values do not match!"
    )

else:

    print(
        "Ridge and XGBoost actual values match."
    )


if not ridge_lstm_match:

    print(
        "WARNING: Ridge and LSTM actual values do not match!"
    )

else:

    print(
        "Ridge and LSTM actual values match."
    )


# ============================================================
# 9. USE COMMON ACTUAL TEMPERATURE
# ============================================================

ensemble_df["actual_temperature"] = (
    ensemble_df["actual_temperature"]
)


# ============================================================
# 10. ENSEMBLE WEIGHTS
# ============================================================

RIDGE_WEIGHT = 1 / 3
XGB_WEIGHT = 1 / 3
LSTM_WEIGHT = 1 / 3


print("\n==========================================")
print("ENSEMBLE WEIGHTS")
print("==========================================")

print(
    f"Ridge   : {RIDGE_WEIGHT:.4f}"
)

print(
    f"XGBoost : {XGB_WEIGHT:.4f}"
)

print(
    f"LSTM    : {LSTM_WEIGHT:.4f}"
)


# ============================================================
# 11. CREATE ENSEMBLE PREDICTION
# ============================================================

ensemble_df["ensemble_prediction"] = (

    RIDGE_WEIGHT *
    ensemble_df["ridge_prediction"]

    +

    XGB_WEIGHT *
    ensemble_df["xgboost_prediction"]

    +

    LSTM_WEIGHT *
    ensemble_df["lstm_prediction"]
)


# ============================================================
# 12. ACTUAL VALUES
# ============================================================

y_actual = ensemble_df[
    "actual_temperature"
]


# ============================================================
# 13. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(actual, prediction):

    mae = mean_absolute_error(
        actual,
        prediction
    )

    mse = mean_squared_error(
        actual,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        prediction
    )

    return mae, mse, rmse, r2


# ============================================================
# 14. RIDGE METRICS
# ============================================================

ridge_mae, ridge_mse, ridge_rmse, ridge_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["ridge_prediction"]
    )
)


# ============================================================
# 15. XGBOOST METRICS
# ============================================================

xgb_mae, xgb_mse, xgb_rmse, xgb_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["xgboost_prediction"]
    )
)


# ============================================================
# 16. LSTM METRICS
# ============================================================

lstm_mae, lstm_mse, lstm_rmse, lstm_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["lstm_prediction"]
    )
)


# ============================================================
# 17. ENSEMBLE METRICS
# ============================================================

ensemble_mae, ensemble_mse, ensemble_rmse, ensemble_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["ensemble_prediction"]
    )
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("TEMPERATURE MODEL RESULTS")
print("==========================================")

print("\nRidge")
print("------------------------------------------")
print(f"MAE  : {ridge_mae:.4f}")
print(f"MSE  : {ridge_mse:.4f}")
print(f"RMSE : {ridge_rmse:.4f}")
print(f"R²   : {ridge_r2:.4f}")


print("\nXGBoost")
print("------------------------------------------")
print(f"MAE  : {xgb_mae:.4f}")
print(f"MSE  : {xgb_mse:.4f}")
print(f"RMSE : {xgb_rmse:.4f}")
print(f"R²   : {xgb_r2:.4f}")


print("\nLSTM")
print("------------------------------------------")
print(f"MAE  : {lstm_mae:.4f}")
print(f"MSE  : {lstm_mse:.4f}")
print(f"RMSE : {lstm_rmse:.4f}")
print(f"R²   : {lstm_r2:.4f}")


print("\nENSEMBLE")
print("------------------------------------------")
print(f"MAE  : {ensemble_mae:.4f}")
print(f"MSE  : {ensemble_mse:.4f}")
print(f"RMSE : {ensemble_rmse:.4f}")
print(f"R²   : {ensemble_r2:.4f}")


# ============================================================
# 19. CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({

    "Model": [
        "Ridge",
        "XGBoost",
        "LSTM",
        "Ensemble"
    ],

    "MAE": [
        ridge_mae,
        xgb_mae,
        lstm_mae,
        ensemble_mae
    ],

    "MSE": [
        ridge_mse,
        xgb_mse,
        lstm_mse,
        ensemble_mse
    ],

    "RMSE": [
        ridge_rmse,
        xgb_rmse,
        lstm_rmse,
        ensemble_rmse
    ],

    "R2": [
        ridge_r2,
        xgb_r2,
        lstm_r2,
        ensemble_r2
    ]

})


# ============================================================
# 20. SAVE ENSEMBLE PREDICTIONS
# ============================================================

prediction_output = os.path.join(
    MODEL_FOLDER,
    "temperature_ensemble_predictions.csv"
)

ensemble_df.to_csv(
    prediction_output,
    index=False
)


# ============================================================
# 21. SAVE MODEL COMPARISON
# ============================================================

comparison_output = os.path.join(
    MODEL_FOLDER,
    "temperature_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_output,
    index=False
)


# ============================================================
# 22. SAVE WEIGHTS
# ============================================================

weights = {

    "ridge": RIDGE_WEIGHT,

    "xgboost": XGB_WEIGHT,

    "lstm": LSTM_WEIGHT

}


weights_output = os.path.join(
    MODEL_FOLDER,
    "temperature_ensemble_weights.pkl"
)

joblib.dump(
    weights,
    weights_output
)


# ============================================================
# 23. SAVE METRICS
# ============================================================

metrics = {

    "ridge": {
        "MAE": ridge_mae,
        "MSE": ridge_mse,
        "RMSE": ridge_rmse,
        "R2": ridge_r2
    },

    "xgboost": {
        "MAE": xgb_mae,
        "MSE": xgb_mse,
        "RMSE": xgb_rmse,
        "R2": xgb_r2
    },

    "lstm": {
        "MAE": lstm_mae,
        "MSE": lstm_mse,
        "RMSE": lstm_rmse,
        "R2": lstm_r2
    },

    "ensemble": {
        "MAE": ensemble_mae,
        "MSE": ensemble_mse,
        "RMSE": ensemble_rmse,
        "R2": ensemble_r2
    }

}


metrics_output = os.path.join(
    MODEL_FOLDER,
    "temperature_ensemble_metrics.pkl"
)

joblib.dump(
    metrics,
    metrics_output
)


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("FILES SAVED")
print("==========================================")

print(
    prediction_output
)

print(
    comparison_output
)

print(
    weights_output
)

print(
    metrics_output
)

print("\nEnsemble completed successfully!")