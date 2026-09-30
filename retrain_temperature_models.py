import os
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/temperature_dataset.csv"
MODEL_FOLDER = "Temperature_Models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "temperature_target_next_day"

TRAIN_RATIO = 0.80

SEQUENCE_LENGTH = 24

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\n==========================================")
print("LOADING TEMPERATURE DATASET")
print("==========================================")

df = pd.read_csv(DATA_PATH)

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values(
    ["city", "datetime"]
).reset_index(drop=True)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES
# ============================================================

excluded_columns = [
    "datetime",
    "city",
    TARGET
]

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in excluded_columns
]


# ============================================================
# CLEAN DATA
# ============================================================

df[FEATURE_COLUMNS] = df[
    FEATURE_COLUMNS
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
    subset=FEATURE_COLUMNS + [TARGET]
).reset_index(drop=True)


print("Number of features:", len(FEATURE_COLUMNS))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n==========================================")
print("CREATING TRAIN / TEST SPLIT")
print("==========================================")

train_parts = []
test_parts = []

cities = df["city"].unique()

for city in cities:

    city_df = df[
        df["city"] == city
    ].copy()

    city_df = city_df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    split_index = int(
        len(city_df) * TRAIN_RATIO
    )

    train_parts.append(
        city_df.iloc[:split_index]
    )

    test_parts.append(
        city_df.iloc[split_index:]
    )

    print(
        f"{city}: "
        f"{split_index} training | "
        f"{len(city_df) - split_index} testing"
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


print("\nTotal training rows:", len(train_df))
print("Total testing rows:", len(test_df))


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ============================================================
# LOAD OLD METRICS
# ============================================================

print("\n==========================================")
print("LOADING OLD MODEL METRICS")
print("==========================================")

old_metrics = {}


metric_files = {
    "Ridge":
        "temperature_ridge_metrics.pkl",

    "XGBoost":
        "temperature_xgboost_metrics.pkl",

    "LSTM":
        "temperature_lstm_metrics.pkl"
}


for model_name, filename in metric_files.items():

    path = os.path.join(
        MODEL_FOLDER,
        filename
    )

    if os.path.exists(path):

        try:

            old_metrics[model_name] = joblib.load(
                path
            )

        except Exception:

            with open(path, "rb") as file:

                old_metrics[model_name] = pickle.load(
                    file
                )

        print(
            f"{model_name}:",
            old_metrics[model_name]
        )

    else:

        old_metrics[model_name] = None

        print(
            f"{model_name}: OLD METRICS NOT FOUND"
        )


# ============================================================
# NEW METRICS STORAGE
# ============================================================

new_metrics = {}


# ============================================================
# 1. RIDGE
# ============================================================

print("\n==========================================")
print("TRAINING NEW RIDGE MODEL")
print("==========================================")

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    TARGET
]

X_test = test_df[
    FEATURE_COLUMNS
]

y_test = test_df[
    TARGET
]


ridge_scaler = StandardScaler()

X_train_scaled = ridge_scaler.fit_transform(
    X_train
)

X_test_scaled = ridge_scaler.transform(
    X_test
)


ridge_model = Ridge(
    alpha=1.0
)


ridge_model.fit(
    X_train_scaled,
    y_train
)


ridge_pred = ridge_model.predict(
    X_test_scaled
)


ridge_metrics = calculate_metrics(
    y_test,
    ridge_pred
)


new_metrics["Ridge"] = ridge_metrics


print("\nNEW RIDGE RESULTS")

for key, value in ridge_metrics.items():

    print(
        f"{key}: {value:.6f}"
    )


# ============================================================
# 2. XGBOOST
# ============================================================

print("\n==========================================")
print("TRAINING NEW XGBOOST MODEL")
print("==========================================")


xgb_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_pred = xgb_model.predict(
    X_test
)


xgb_metrics = calculate_metrics(
    y_test,
    xgb_pred
)


new_metrics["XGBoost"] = xgb_metrics


print("\nNEW XGBOOST RESULTS")

for key, value in xgb_metrics.items():

    print(
        f"{key}: {value:.6f}"
    )


# ============================================================
# 3. LSTM DATA
# ============================================================

print("\n==========================================")
print("PREPARING LSTM DATA")
print("==========================================")


lstm_scaler = StandardScaler()

lstm_scaler.fit(
    train_df[FEATURE_COLUMNS]
)


def create_training_sequences():

    X = []
    y = []

    for city in cities:

        city_df = train_df[
            train_df["city"] == city
        ].sort_values(
            "datetime"
        ).reset_index(drop=True)

        scaled = lstm_scaler.transform(
            city_df[FEATURE_COLUMNS]
        )

        targets = city_df[
            TARGET
        ].values

        for i in range(
            SEQUENCE_LENGTH,
            len(city_df)
        ):

            X.append(
                scaled[
                    i - SEQUENCE_LENGTH:i
                ]
            )

            y.append(
                targets[i]
            )

    return (
        np.array(X),
        np.array(y)
    )


X_lstm_train, y_lstm_train = (
    create_training_sequences()
)


# ============================================================
# LSTM TEST SEQUENCES
# ============================================================

def create_test_sequences():

    X = []
    y = []

    for city in cities:

        city_train = train_df[
            train_df["city"] == city
        ].sort_values(
            "datetime"
        ).reset_index(drop=True)

        city_test = test_df[
            test_df["city"] == city
        ].sort_values(
            "datetime"
        ).reset_index(drop=True)

        history = city_train.tail(
            SEQUENCE_LENGTH
        )

        combined = pd.concat(
            [
                history,
                city_test
            ],
            ignore_index=True
        )

        scaled = lstm_scaler.transform(
            combined[FEATURE_COLUMNS]
        )

        targets = city_test[
            TARGET
        ].values

        for i in range(
            len(city_test)
        ):

            X.append(
                scaled[
                    i:i + SEQUENCE_LENGTH
                ]
            )

            y.append(
                targets[i]
            )

    return (
        np.array(X),
        np.array(y)
    )


X_lstm_test, y_lstm_test = (
    create_test_sequences()
)


print(
    "LSTM training shape:",
    X_lstm_train.shape
)

print(
    "LSTM testing shape:",
    X_lstm_test.shape
)


# ============================================================
# BUILD NEW LSTM
# ============================================================

print("\n==========================================")
print("TRAINING NEW LSTM MODEL")
print("==========================================")


lstm_model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        )
    ),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1
    )
])


lstm_model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


lstm_model.fit(

    X_lstm_train,

    y_lstm_train,

    validation_split=0.10,

    epochs=30,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# LSTM PREDICTIONS
# ============================================================

lstm_pred = lstm_model.predict(
    X_lstm_test,
    verbose=0
).flatten()


lstm_metrics = calculate_metrics(
    y_lstm_test,
    lstm_pred
)


new_metrics["LSTM"] = lstm_metrics


print("\nNEW LSTM RESULTS")

for key, value in lstm_metrics.items():

    print(
        f"{key}: {value:.6f}"
    )


# ============================================================
# COMPARE OLD VS NEW
# ============================================================

print("\n==========================================")
print("OLD VS NEW MODEL COMPARISON")
print("==========================================")


comparison_rows = []


for model_name in [
    "Ridge",
    "XGBoost",
    "LSTM"
]:

    old = old_metrics.get(
        model_name
    )

    new = new_metrics.get(
        model_name
    )

    if old is None:

        comparison_rows.append({

            "model": model_name,

            "old_MAE": np.nan,
            "new_MAE": new["MAE"],

            "old_RMSE": np.nan,
            "new_RMSE": new["RMSE"],

            "old_R2": np.nan,
            "new_R2": new["R2"],

            "improved": "NO OLD MODEL"

        })

        continue


    # Lower MAE and RMSE are better.
    # Higher R2 is better.

    mae_improved = (
        new["MAE"] < old["MAE"]
    )

    rmse_improved = (
        new["RMSE"] < old["RMSE"]
    )

    r2_improved = (
        new["R2"] > old["R2"]
    )


    # Require all three primary metrics
    # to improve before automatically
    # considering the new model better.

    improved = (
        mae_improved
        and
        rmse_improved
        and
        r2_improved
    )


    comparison_rows.append({

        "model": model_name,

        "old_MAE": old["MAE"],
        "new_MAE": new["MAE"],

        "old_RMSE": old["RMSE"],
        "new_RMSE": new["RMSE"],

        "old_R2": old["R2"],
        "new_R2": new["R2"],

        "improved": improved

    })


comparison_df = pd.DataFrame(
    comparison_rows
)


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    MODEL_FOLDER,
    "temperature_retraining_comparison.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE NEW MODELS TO TEMPORARY FILES
# ============================================================

print("\n==========================================")
print("SAVING NEW MODELS FOR INSPECTION")
print("==========================================")


# IMPORTANT:
# These are NOT replacing your current models.

ridge_new_path = os.path.join(
    MODEL_FOLDER,
    "temperature_ridge_model_NEW.pkl"
)

with open(
    ridge_new_path,
    "wb"
) as file:

    pickle.dump(
        ridge_model,
        file
    )


ridge_scaler_new_path = os.path.join(
    MODEL_FOLDER,
    "temperature_ridge_scaler_NEW.pkl"
)

with open(
    ridge_scaler_new_path,
    "wb"
) as file:

    pickle.dump(
        ridge_scaler,
        file
    )


xgb_new_path = os.path.join(
    MODEL_FOLDER,
    "temperature_xgboost_model_NEW.pkl"
)

joblib.dump(
    xgb_model,
    xgb_new_path
)


lstm_new_path = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_model_NEW.keras"
)

lstm_model.save(
    lstm_new_path
)


lstm_scaler_new_path = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_scaler_NEW.pkl"
)

joblib.dump(
    lstm_scaler,
    lstm_scaler_new_path
)


# ============================================================
# SAVE NEW METRICS
# ============================================================

new_metrics_path = os.path.join(
    MODEL_FOLDER,
    "temperature_new_metrics.pkl"
)

joblib.dump(
    new_metrics,
    new_metrics_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("RETRAINING COMPLETE")
print("==========================================")

print("\nExisting models were NOT replaced.")

print("\nComparison:")
print(
    comparison_path
)

print("\nNew models:")
print(
    ridge_new_path
)

print(
    xgb_new_path
)

print(
    lstm_new_path
)

print("\nNew metrics:")
print(
    new_metrics_path
)

print("\n==========================================")
print("NEXT STEP")
print("==========================================")

print(
    "Review the OLD vs NEW comparison first."
)

print(
    "Only after checking the results should "
    "we replace any existing model."
)