import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "temperature_dataset.csv"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "Temperature_Models",
    "temperature_lstm_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "Temperature_Models",
    "temperature_lstm_features.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading temperature dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# LOAD THE EXACT FEATURES USED BY LSTM
# ============================================================

print("Loading LSTM features...")

with open(FEATURES_PATH, "rb") as file:

    features = pickle.load(file)


print(
    f"Number of features: {len(features)}"
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Some LSTM features are missing from the dataset."
    )


# ============================================================
# PREPARE DATA
# ============================================================

X = df[features].copy()


# Handle infinity
X = X.replace(
    [float("inf"), float("-inf")],
    float("nan")
)


# Fill missing values using training-dataset medians
X = X.fillna(
    X.median()
)


# ============================================================
# CREATE SCALER
# ============================================================

print("Creating StandardScaler...")

scaler = StandardScaler()

scaler.fit(X)


# ============================================================
# SAVE CORRECTLY
# ============================================================

print("Saving scaler...")

with open(
    SCALER_PATH,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# VERIFY
# ============================================================

print("Verifying saved scaler...")

with open(
    SCALER_PATH,
    "rb"
) as file:

    test_scaler = pickle.load(file)


print()
print("=" * 45)
print("LSTM SCALER FIXED SUCCESSFULLY")
print("=" * 45)

print(
    f"Features : {len(features)}"
)

print(
    f"Scaler   : {SCALER_PATH}"
)

print(
    f"File size: "
    f"{os.path.getsize(SCALER_PATH)} bytes"
)

print(
    f"Mean shape: {test_scaler.mean_.shape}"
)

print(
    f"Scale shape: {test_scaler.scale_.shape}"
)

print("=" * 45)