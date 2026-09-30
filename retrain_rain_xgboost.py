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
# SETTINGS
# ============================================================

DATA_FILE = "data/rain_dataset.csv"

MODEL_FOLDER = "Rain_Models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_model.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_features.pkl"
)

METRICS_FILE = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_metrics.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_FOLDER,
    "rain_xgboost_test_predictions.csv"
)

TARGET = "rain_target_next_day"

TRAIN_RATIO = 0.80


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("RAIN XGBOOST RETRAINING")
print("==========================================")

print("\nLoading rain dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# BASIC CLEANING
# ============================================================

if "datetime" in df.columns:

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )


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
# CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ============================================================
# SEPARATE TARGET
# ============================================================

y = df[TARGET].copy()

X = df.drop(
    columns=[TARGET]
)


# ============================================================
# KEEP IDENTIFIERS
# ============================================================

identifier_columns = []

if "datetime" in X.columns:
    identifier_columns.append("datetime")

if "city" in X.columns:
    identifier_columns.append("city")


identifiers = X[
    identifier_columns
].copy()


# ============================================================
# REMOVE IDENTIFIERS FROM FEATURES
# ============================================================

X = X.drop(
    columns=identifier_columns,
    errors="ignore"
)


# ============================================================
# KEEP ONLY NUMERIC FEATURES
# ============================================================

X = X.select_dtypes(
    include=[np.number]
)


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(
    X.median()
)


# ============================================================
# REMOVE ROWS WITH INVALID TARGET
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
# FEATURE LIST
# ============================================================

features = X.columns.tolist()


print("\nNumber of features:", len(features))

print("\nFeatures:")

for feature in features:
    print(" -", feature)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(X) * TRAIN_RATIO
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

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# TRAIN NEW XGBOOST MODEL
# ============================================================

print("\n==========================================")
print("TRAINING NEW XGBOOST MODEL")
print("==========================================")


new_model = XGBRegressor(

    n_estimators=500,

    max_depth=8,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


new_model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ============================================================
# NEW MODEL PREDICTIONS
# ============================================================

print("\nGenerating new predictions...")


new_predictions = new_model.predict(
    X_test
)


# ============================================================
# NEW MODEL METRICS
# ============================================================

new_mae = mean_absolute_error(
    y_test,
    new_predictions
)

new_mse = mean_squared_error(
    y_test,
    new_predictions
)

new_rmse = np.sqrt(
    new_mse
)

new_r2 = r2_score(
    y_test,
    new_predictions
)


new_metrics = {

    "MAE": float(new_mae),

    "MSE": float(new_mse),

    "RMSE": float(new_rmse),

    "R2": float(new_r2)
}


# ============================================================
# DISPLAY NEW RESULTS
# ============================================================

print("\n==========================================")
print("NEW MODEL RESULTS")
print("==========================================")

print(f"MAE  : {new_mae:.4f}")
print(f"MSE  : {new_mse:.4f}")
print(f"RMSE : {new_rmse:.4f}")
print(f"R²   : {new_r2:.4f}")


# ============================================================
# LOAD OLD MODEL METRICS
# ============================================================

old_metrics = None

if os.path.exists(METRICS_FILE):

    try:

        old_metrics = joblib.load(
            METRICS_FILE
        )

        print("\n==========================================")
        print("CURRENT MODEL RESULTS")
        print("==========================================")

        print(
            f"MAE  : {old_metrics['MAE']:.4f}"
        )

        print(
            f"MSE  : {old_metrics['MSE']:.4f}"
        )

        print(
            f"RMSE : {old_metrics['RMSE']:.4f}"
        )

        print(
            f"R²   : {old_metrics['R2']:.4f}"
        )

    except Exception as e:

        print(
            "\nCould not load existing metrics."
        )

        print(
            "Reason:",
            e
        )

        old_metrics = None


# ============================================================
# COMPARE MODELS
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")


if old_metrics is None:

    print(
        "No previous model metrics found."
    )

    should_replace = True

else:

    # --------------------------------------------------------
    # Primary metric: RMSE
    #
    # Lower RMSE = better
    # --------------------------------------------------------

    old_rmse = old_metrics["RMSE"]

    print(
        f"Old RMSE : {old_rmse:.6f}"
    )

    print(
        f"New RMSE : {new_rmse:.6f}"
    )


    if new_rmse < old_rmse:

        should_replace = True

        print(
            "\nNEW MODEL IS BETTER."
        )

        print(
            "The existing model will be replaced."
        )

    else:

        should_replace = False

        print(
            "\nEXISTING MODEL IS BETTER OR EQUAL."
        )

        print(
            "The existing model will be kept."
        )


# ============================================================
# SAVE ONLY IF NEW MODEL IS BETTER
# ============================================================

if should_replace:

    print("\n==========================================")
    print("SAVING NEW MODEL")
    print("==========================================")


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        new_model,
        MODEL_FILE
    )


    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    joblib.dump(
        features,
        FEATURE_FILE
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    joblib.dump(
        new_metrics,
        METRICS_FILE
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = identifiers_test.copy()


    prediction_df["actual_rain"] = (
        y_test.values
    )


    prediction_df["xgboost_prediction"] = (
        new_predictions
    )


    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False
    )


    print("Model saved:")
    print(MODEL_FILE)

    print("Features saved:")
    print(FEATURE_FILE)

    print("Metrics saved:")
    print(METRICS_FILE)

    print("Predictions saved:")
    print(PREDICTION_FILE)


# ============================================================
# KEEP OLD MODEL
# ============================================================

else:

    print("\n==========================================")
    print("MODEL NOT REPLACED")
    print("==========================================")

    print(
        "The existing Rain XGBoost model is still being used."
    )

    print(
        "\nNo existing model files were overwritten."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("RETRAINING COMPLETED")
print("==========================================")

if should_replace:

    print(
        "Result: NEW MODEL ACCEPTED"
    )

else:

    print(
        "Result: OLD MODEL RETAINED"
    )