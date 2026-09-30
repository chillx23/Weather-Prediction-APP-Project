import os
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_FILE = "data/rain_dataset.csv"

MODEL_FOLDER = "Rain_Models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading rain dataset...")

df = pd.read_csv(DATA_FILE)

print("Rain dataset loaded!")
print("Shape:", df.shape)


# ============================================================
# 3. BASIC CLEANING
# ============================================================

# Convert datetime if it exists
if "datetime" in df.columns:

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )


# Remove completely empty rows
df = df.dropna(
    how="all"
)


# Sort chronologically
if "datetime" in df.columns:

    df = df.sort_values(
        ["city", "datetime"]
    ).reset_index(
        drop=True
    )


# ============================================================
# 4. TARGET COLUMN
# ============================================================

TARGET = "rain_target_next_day"


if TARGET not in df.columns:

    raise ValueError(
        f"\nTarget column '{TARGET}' was not found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ============================================================
# 5. REMOVE TARGET FROM FEATURES
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# 6. REMOVE NON-NUMERIC COLUMNS
# ============================================================

# Save useful identifying columns
identifier_columns = []

if "datetime" in X.columns:
    identifier_columns.append("datetime")

if "city" in X.columns:
    identifier_columns.append("city")


# Keep identifiers separately
identifiers = X[
    identifier_columns
].copy()


# Remove identifiers from model features
X = X.drop(
    columns=identifier_columns,
    errors="ignore"
)


# Keep only numeric features
X = X.select_dtypes(
    include=[np.number]
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# 8. REMOVE ROWS WITH MISSING TARGET
# ============================================================

valid_rows = y.notna()

X = X.loc[
    valid_rows
].reset_index(
    drop=True
)

y = y.loc[
    valid_rows
].reset_index(
    drop=True
)

identifiers = identifiers.loc[
    valid_rows
].reset_index(
    drop=True
)


# ============================================================
# 9. FEATURE LIST
# ============================================================

features = X.columns.tolist()


print("\n==========================================")
print("RAIN FEATURES")
print("==========================================")

print(
    "Number of features:",
    len(features)
)

print("\nFeatures:")

for feature in features:

    print(
        feature
    )


# ============================================================
# 10. CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

# IMPORTANT:
# Do not randomly shuffle weather data.

split_index = int(
    len(X) * 0.80
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]


y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


identifiers_test = identifiers.iloc[
    split_index:
].copy()


print("\n==========================================")
print("DATA SPLIT")
print("==========================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 11. CREATE XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")


model = XGBRegressor(

    n_estimators=500,

    max_depth=8,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1

)


# ============================================================
# 12. TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


print(
    "Training completed!"
)


# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")


predictions = model.predict(
    X_test
)


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)


mse = mean_squared_error(
    y_test,
    predictions
)


rmse = np.sqrt(
    mse
)


r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("XGBOOST RAIN RESULTS")
print("==========================================")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"MSE  : {mse:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 16. CREATE PREDICTION DATAFRAME
# ============================================================

prediction_df = identifiers_test.copy()


prediction_df["actual_rain"] = (
    y_test.values
)


prediction_df["xgboost_prediction"] = (
    predictions
)


# ============================================================
# 17. DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")

print(
    prediction_df.head(20)
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_model.pkl"
)


joblib.dump(
    model,
    model_path
)


# ============================================================
# 19. SAVE FEATURES
# ============================================================

features_path = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_features.pkl"
)


joblib.dump(
    features,
    features_path
)


# ============================================================
# 20. SAVE METRICS
# ============================================================

metrics = {

    "MAE": float(mae),

    "MSE": float(mse),

    "RMSE": float(rmse),

    "R2": float(r2)

}


metrics_path = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_metrics.pkl"
)


joblib.dump(
    metrics,
    metrics_path
)


# ============================================================
# 21. SAVE TEST PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_test_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("FILES SAVED")
print("==========================================")

print(
    "Model:",
    model_path
)

print(
    "Features:",
    features_path
)

print(
    "Metrics:",
    metrics_path
)

print(
    "Predictions:",
    prediction_path
)

print("\nRain XGBoost model completed successfully!")