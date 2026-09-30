import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/temperature_dataset.csv"

MODEL_FOLDER = "Temperature_Models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_model.keras"
)

SCALER_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_scaler.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_features.pkl"
)

SEQUENCE_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_sequence_length.pkl"
)

METRICS_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_metrics.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_FOLDER,
    "temperature_lstm_test_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 24

TRAIN_RATIO = 0.80

TARGET = "temperature_target_next_day"


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
print("LSTM TEMPERATURE RETRAINING")
print("==========================================")

print("\nLoading temperature dataset...")

df = pd.read_csv(
    DATA_PATH
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

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in [
        "datetime",
        "city",
        TARGET
    ]
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
).reset_index(
    drop=True
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_parts = []
test_parts = []

cities = df["city"].unique()


for city in cities:

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

    train_parts.append(
        city_df.iloc[:split_index]
    )

    test_parts.append(
        city_df.iloc[split_index:]
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


print("\nTraining rows:", len(train_df))
print("Testing rows :", len(test_df))


# ============================================================
# SCALER
# ============================================================

scaler = StandardScaler()

scaler.fit(
    train_df[FEATURE_COLUMNS]
)


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

X_train = []
y_train = []


for city in cities:

    city_df = train_df[
        train_df["city"] == city
    ].sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    scaled_features = scaler.transform(
        city_df[FEATURE_COLUMNS]
    )

    targets = city_df[
        TARGET
    ].values

    for i in range(
        SEQUENCE_LENGTH,
        len(city_df)
    ):

        X_train.append(
            scaled_features[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y_train.append(
            targets[i]
        )


X_train = np.array(
    X_train
)

y_train = np.array(
    y_train
)


# ============================================================
# CREATE TESTING SEQUENCES
# ============================================================

X_test = []
y_test = []

test_datetimes = []
test_cities = []


for city in cities:

    city_train = train_df[
        train_df["city"] == city
    ].sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    city_test = test_df[
        test_df["city"] == city
    ].sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

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

    scaled_features = scaler.transform(
        combined[FEATURE_COLUMNS]
    )

    targets = city_test[
        TARGET
    ].values

    for i in range(
        len(city_test)
    ):

        sequence = scaled_features[
            i:i + SEQUENCE_LENGTH
        ]

        X_test.append(
            sequence
        )

        y_test.append(
            targets[i]
        )

        test_datetimes.append(
            city_test.loc[
                i,
                "datetime"
            ]
        )

        test_cities.append(
            city_test.loc[
                i,
                "city"
            ]
        )


X_test = np.array(
    X_test
)

y_test = np.array(
    y_test
)

test_datetimes = np.array(
    test_datetimes
)

test_cities = np.array(
    test_cities
)


print("\n==========================================")
print("LSTM DATA")
print("==========================================")

print(
    "Training sequence shape:",
    X_train.shape
)

print(
    "Testing sequence shape:",
    X_test.shape
)


# ============================================================
# BUILD NEW MODEL
# ============================================================

print("\n==========================================")
print("BUILDING NEW LSTM MODEL")
print("==========================================")


new_model = Sequential([

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


# ============================================================
# COMPILE
# ============================================================

new_model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n==========================================")
print("TRAINING NEW LSTM")
print("==========================================")


new_model.fit(

    X_train,

    y_train,

    validation_split=0.10,

    epochs=30,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


new_predictions = new_model.predict(
    X_test,
    verbose=0
).flatten()


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
print("NEW LSTM RESULTS")
print("==========================================")

print(
    f"MAE  : {new_mae:.4f}"
)

print(
    f"MSE  : {new_mse:.4f}"
)

print(
    f"RMSE : {new_rmse:.4f}"
)

print(
    f"R²   : {new_r2:.4f}"
)


# ============================================================
# LOAD OLD METRICS
# ============================================================

old_metrics = None


if os.path.exists(METRICS_FILE):

    try:

        old_metrics = joblib.load(
            METRICS_FILE
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
            "\nNEW LSTM MODEL IS BETTER."
        )

        print(
            "Existing model will be replaced."
        )

    else:

        should_replace = False

        print(
            "\nEXISTING LSTM MODEL IS BETTER OR EQUAL."
        )

        print(
            "Existing model will be kept."
        )


# ============================================================
# SAVE ONLY IF BETTER
# ============================================================

if should_replace:

    print("\n==========================================")
    print("SAVING NEW LSTM MODEL")
    print("==========================================")


    # Model
    new_model.save(
        MODEL_FILE
    )


    # Scaler
    joblib.dump(
        scaler,
        SCALER_FILE
    )


    # Features
    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_FILE
    )


    # Sequence length
    joblib.dump(
        SEQUENCE_LENGTH,
        SEQUENCE_FILE
    )


    # Metrics
    joblib.dump(
        new_metrics,
        METRICS_FILE
    )


    # Predictions
    prediction_df = pd.DataFrame({

        "datetime":
            test_datetimes,

        "city":
            test_cities,

        TARGET:
            y_test,

        "lstm_prediction":
            new_predictions

    })


    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False
    )


    print("New LSTM model saved.")
    print("Model:", MODEL_FILE)
    print("Scaler:", SCALER_FILE)
    print("Features:", FEATURE_FILE)
    print("Sequence:", SEQUENCE_FILE)
    print("Metrics:", METRICS_FILE)
    print("Predictions:", PREDICTION_FILE)


else:

    print("\n==========================================")
    print("LSTM MODEL NOT REPLACED")
    print("==========================================")

    print(
        "The existing LSTM model remains active."
    )


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("LSTM RETRAINING COMPLETED")
print("==========================================")

if should_replace:

    print(
        "Result: NEW MODEL ACCEPTED"
    )

else:

    print(
        "Result: OLD MODEL RETAINED"
    )