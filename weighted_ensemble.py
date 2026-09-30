import os
import numpy as np
import pandas as pd
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
# 2. MANUAL ENSEMBLE WEIGHTS
# ============================================================

RIDGE_WEIGHT = 0.218305
XGB_WEIGHT = 0.720254
LSTM_WEIGHT = 0.061441


# Check that weights add up to 1
TOTAL_WEIGHT = (
    RIDGE_WEIGHT
    + XGB_WEIGHT
    + LSTM_WEIGHT
)


if not np.isclose(TOTAL_WEIGHT, 1.0):
    raise ValueError(
        f"Ensemble weights must add up to 1. "
        f"Current total = {TOTAL_WEIGHT}"
    )


# ============================================================
# 3. LOAD PREDICTION FILES
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
# 4. CONVERT DATETIME
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
# 5. PREPARE RIDGE DATA
# ============================================================

ridge_df = ridge_df[
    [
        "datetime",
        "city",
        "temperature_target_next_day",
        "ridge_prediction"
    ]
].copy()


ridge_df = ridge_df.rename(
    columns={
        "temperature_target_next_day":
            "actual_temperature"
    }
)


# ============================================================
# 6. PREPARE XGBOOST DATA
# ============================================================

xgb_df = xgb_df[
    [
        "datetime",
        "city",
        "temperature_target_next_day",
        "xgboost_prediction"
    ]
].copy()


xgb_df = xgb_df.rename(
    columns={
        "temperature_target_next_day":
            "actual_temperature_xgb"
    }
)


# ============================================================
# 7. PREPARE LSTM DATA
# ============================================================

lstm_df = lstm_df[
    [
        "datetime",
        "city",
        "temperature_target_next_day",
        "lstm_prediction"
    ]
].copy()


lstm_df = lstm_df.rename(
    columns={
        "temperature_target_next_day":
            "actual_temperature_lstm"
    }
)


# ============================================================
# 8. MERGE RIDGE + XGBOOST
# ============================================================

ensemble_df = pd.merge(
    ridge_df,
    xgb_df,
    on=[
        "datetime",
        "city"
    ],
    how="inner"
)


# ============================================================
# 9. MERGE LSTM
# ============================================================

ensemble_df = pd.merge(
    ensemble_df,
    lstm_df,
    on=[
        "datetime",
        "city"
    ],
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
# 10. CHECK ACTUAL VALUES
# ============================================================

print("\nChecking actual temperature values...")


ridge_actual = ensemble_df[
    "actual_temperature"
].values


xgb_actual = ensemble_df[
    "actual_temperature_xgb"
].values


lstm_actual = ensemble_df[
    "actual_temperature_lstm"
].values


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


if ridge_xgb_match:

    print(
        "Ridge and XGBoost actual values match."
    )

else:

    print(
        "WARNING: Ridge and XGBoost actual values "
        "DO NOT match!"
    )


if ridge_lstm_match:

    print(
        "Ridge and LSTM actual values match."
    )

else:

    print(
        "WARNING: Ridge and LSTM actual values "
        "DO NOT match!"
    )


# ============================================================
# 11. DISPLAY MANUAL WEIGHTS
# ============================================================

print("\n==========================================")
print("MANUAL ENSEMBLE WEIGHTS")
print("==========================================")

print(
    f"Ridge   : {RIDGE_WEIGHT:.6f} "
    f"({RIDGE_WEIGHT * 100:.2f}%)"
)

print(
    f"XGBoost : {XGB_WEIGHT:.6f} "
    f"({XGB_WEIGHT * 100:.2f}%)"
)

print(
    f"LSTM    : {LSTM_WEIGHT:.6f} "
    f"({LSTM_WEIGHT * 100:.2f}%)"
)

print(
    f"Total   : {TOTAL_WEIGHT:.6f}"
)


# ============================================================
# 12. CREATE WEIGHTED ENSEMBLE
# ============================================================

ensemble_df["ensemble_prediction"] = (

    RIDGE_WEIGHT
    * ensemble_df["ridge_prediction"]

    +

    XGB_WEIGHT
    * ensemble_df["xgboost_prediction"]

    +

    LSTM_WEIGHT
    * ensemble_df["lstm_prediction"]
)


# ============================================================
# 13. ACTUAL TEMPERATURE
# ============================================================

y_actual = ensemble_df[
    "actual_temperature"
].values


# ============================================================
# 14. METRICS FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    prediction
):

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

    return (
        mae,
        mse,
        rmse,
        r2
    )


# ============================================================
# 15. RIDGE METRICS
# ============================================================

ridge_mae, ridge_mse, ridge_rmse, ridge_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["ridge_prediction"].values
    )
)


# ============================================================
# 16. XGBOOST METRICS
# ============================================================

xgb_mae, xgb_mse, xgb_rmse, xgb_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["xgboost_prediction"].values
    )
)


# ============================================================
# 17. LSTM METRICS
# ============================================================

lstm_mae, lstm_mse, lstm_rmse, lstm_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["lstm_prediction"].values
    )
)


# ============================================================
# 18. ENSEMBLE METRICS
# ============================================================

ensemble_mae, ensemble_mse, ensemble_rmse, ensemble_r2 = (
    calculate_metrics(
        y_actual,
        ensemble_df["ensemble_prediction"].values
    )
)


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("TEMPERATURE MODEL RESULTS")
print("==========================================")


print("\nRidge")
print("------------------------------------------")

print(
    f"MAE  : {ridge_mae:.4f}"
)

print(
    f"MSE  : {ridge_mse:.4f}"
)

print(
    f"RMSE : {ridge_rmse:.4f}"
)

print(
    f"R²   : {ridge_r2:.4f}"
)


print("\nXGBoost")
print("------------------------------------------")

print(
    f"MAE  : {xgb_mae:.4f}"
)

print(
    f"MSE  : {xgb_mse:.4f}"
)

print(
    f"RMSE : {xgb_rmse:.4f}"
)

print(
    f"R²   : {xgb_r2:.4f}"
)


print("\nLSTM")
print("------------------------------------------")

print(
    f"MAE  : {lstm_mae:.4f}"
)

print(
    f"MSE  : {lstm_mse:.4f}"
)

print(
    f"RMSE : {lstm_rmse:.4f}"
)

print(
    f"R²   : {lstm_r2:.4f}"
)


print("\nWEIGHTED ENSEMBLE")
print("------------------------------------------")

print(
    f"MAE  : {ensemble_mae:.4f}"
)

print(
    f"MSE  : {ensemble_mse:.4f}"
)

print(
    f"RMSE : {ensemble_rmse:.4f}"
)

print(
    f"R²   : {ensemble_r2:.4f}"
)


# ============================================================
# 20. CREATE MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({

    "Model": [
        "Ridge",
        "XGBoost",
        "LSTM",
        "Weighted Ensemble"
    ],

    "Weight": [
        RIDGE_WEIGHT,
        XGB_WEIGHT,
        LSTM_WEIGHT,
        1.0
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
# 21. SAVE ENSEMBLE PREDICTIONS
# ============================================================

prediction_output = os.path.join(
    MODEL_FOLDER,
    "temperature_weighted_ensemble_predictions.csv"
)


ensemble_df.to_csv(
    prediction_output,
    index=False
)


# ============================================================
# 22. SAVE MODEL COMPARISON
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
# 23. SAVE WEIGHTS
# ============================================================

weights = {

    "ridge": RIDGE_WEIGHT,

    "xgboost": XGB_WEIGHT,

    "lstm": LSTM_WEIGHT

}


weights_output = os.path.join(
    MODEL_FOLDER,
    "temperature_manual_weights.pkl"
)


joblib.dump(
    weights,
    weights_output
)


# ============================================================
# 24. SAVE METRICS
# ============================================================

metrics = {

    "ridge": {

        "MAE": float(ridge_mae),
        "MSE": float(ridge_mse),
        "RMSE": float(ridge_rmse),
        "R2": float(ridge_r2)

    },

    "xgboost": {

        "MAE": float(xgb_mae),
        "MSE": float(xgb_mse),
        "RMSE": float(xgb_rmse),
        "R2": float(xgb_r2)

    },

    "lstm": {

        "MAE": float(lstm_mae),
        "MSE": float(lstm_mse),
        "RMSE": float(lstm_rmse),
        "R2": float(lstm_r2)

    },

    "weighted_ensemble": {

        "MAE": float(ensemble_mae),
        "MSE": float(ensemble_mse),
        "RMSE": float(ensemble_rmse),
        "R2": float(ensemble_r2)

    }

}


metrics_output = os.path.join(
    MODEL_FOLDER,
    "temperature_manual_ensemble_metrics.pkl"
)


joblib.dump(
    metrics,
    metrics_output
)


# ============================================================
# 25. SAVE WEIGHTS AS CSV
# ============================================================

weight_table = pd.DataFrame({

    "Model": [
        "Ridge",
        "XGBoost",
        "LSTM"
    ],

    "Weight": [
        RIDGE_WEIGHT,
        XGB_WEIGHT,
        LSTM_WEIGHT
    ],

    "Percentage": [
        RIDGE_WEIGHT * 100,
        XGB_WEIGHT * 100,
        LSTM_WEIGHT * 100
    ]

})


weight_table_output = os.path.join(
    MODEL_FOLDER,
    "temperature_manual_weights.csv"
)


weight_table.to_csv(
    weight_table_output,
    index=False
)


# ============================================================
# 26. FINAL OUTPUT
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

print(
    weight_table_output
)

print("\nWeighted ensemble completed successfully!")