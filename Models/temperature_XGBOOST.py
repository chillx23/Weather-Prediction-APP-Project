import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/temperature_dataset.csv"

MODEL_FOLDER = "Temperature_Models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_xgboost_model.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_xgboost_features.pkl"
)

METRICS_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_xgboost_metrics.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_xgboost_test_predictions.csv"
)

TARGET = "temperature_target_next_day"

TRAIN_RATIO = 0.80


# ============================================================
# CREATE FOLDER
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("==========================================")
print("XGBOOST TEMPERATURE RETRAINING")
print("==========================================")

print("\nLoading temperature dataset...")

df = pd.read_csv(
    INPUT_FILE
)

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = df.sort_values(
    ["city", "datetime"]
).reset_index(
    drop=True
)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES
# ============================================================

excluded_columns = [
    "datetime",
    "city",
    TARGET
]

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


# ============================================================
# CLEAN
# ============================================================

df[feature_columns] = df[
    feature_columns
].replace(
    [np.inf, -np.inf],
    np.nan
)

df[TARGET] = df[
    TARGET
].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=feature_columns + [TARGET]
).reset_index(
    drop=True
)


# ============================================================
# TRAIN / TEST SPLIT PER CITY
# ============================================================

train_data = []
test_data = []


for city in df["city"].unique():

    city_df = df[
        df["city"] == city
    ].copy()

    city_df = city_df.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(city_df) * TRAIN_RATIO
    )

    train_data.append(
        city_df.iloc[:split_index]
    )

    test_data.append(
        city_df.iloc[split_index:]
    )


train_df = pd.concat(
    train_data,
    ignore_index=True
)

test_df = pd.concat(
    test_data,
    ignore_index=True
)


print("\nTraining samples:", len(train_df))
print("Testing samples :", len(test_df))


# ============================================================
# FEATURES / TARGET
# ============================================================

X_train = train_df[
    feature_columns
]

y_train = train_df[
    TARGET
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    TARGET
]


# ============================================================
# TRAIN NEW XGBOOST
# ============================================================

print("\n==========================================")
print("TRAINING NEW XGBOOST MODEL")
print("==========================================")


new_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

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
# PREDICTIONS
# ============================================================

new_predictions = new_model.predict(
    X_test
)


# ============================================================
# METRICS
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


print("\n==========================================")
print("NEW XGBOOST RESULTS")
print("==========================================")

print(f"MAE  : {new_mae:.4f}")
print(f"MSE  : {new_mse:.4f}")
print(f"RMSE : {new_rmse:.4f}")
print(f"R²   : {new_r2:.4f}")


# ============================================================
# LOAD OLD METRICS
# ============================================================

old_metrics = None


if os.path.exists(METRICS_FILE):

    try:

        with open(
            METRICS_FILE,
            "rb"
        ) as file:

            old_metrics = pickle.load(
                file
            )

    except Exception as e:

        print(
            "\nCould not load old metrics:",
            e
        )


# ============================================================
# COMPARE
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")


if old_metrics is None:

    print(
        "No previous metrics found."
    )

    should_replace = True

else:

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
            "\nNEW XGBOOST MODEL IS BETTER."
        )

        print(
            "Existing model will be replaced."
        )

    else:

        should_replace = False

        print(
            "\nEXISTING XGBOOST MODEL IS BETTER OR EQUAL."
        )

        print(
            "Existing model will be kept."
        )


# ============================================================
# SAVE IF BETTER
# ============================================================

if should_replace:

    print("\n==========================================")
    print("SAVING NEW XGBOOST MODEL")
    print("==========================================")


    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            new_model,
            file
        )


    with open(
        FEATURE_FILE,
        "wb"
    ) as file:

        pickle.dump(
            feature_columns,
            file
        )


    with open(
        METRICS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            new_metrics,
            file
        )


    results = test_df[
        [
            "datetime",
            "city",
            TARGET
        ]
    ].copy()

    results["xgboost_prediction"] = (
        new_predictions
    )


    results.to_csv(
        PREDICTION_FILE,
        index=False
    )


    print("New XGBoost model saved.")
    print("Model:", MODEL_FILE)
    print("Features:", FEATURE_FILE)
    print("Metrics:", METRICS_FILE)
    print("Predictions:", PREDICTION_FILE)


else:

    print("\n==========================================")
    print("XGBOOST MODEL NOT REPLACED")
    print("==========================================")

    print(
        "The existing XGBoost model remains active."
    )


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("XGBOOST RETRAINING COMPLETED")
print("==========================================")

if should_replace:
    print("Result: NEW MODEL ACCEPTED")
else:
    print("Result: OLD MODEL RETAINED")