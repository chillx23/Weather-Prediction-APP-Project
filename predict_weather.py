# ============================================================
# predict_weather.py
# Complete weather prediction system
#
# Models:
#   Temperature:
#       - Ridge
#       - XGBoost
#       - LSTM
#       - Weighted ensemble
#
#   Rain:
#       - XGBoost rainfall amount
#       - CNN satellite-image rain/no-rain classification
#
# Uses ONLY relative paths so the project can be shared.
# ============================================================

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import requests

# Suppress the harmless sklearn feature-name warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# ============================================================
# TENSORFLOW
# ============================================================

import tensorflow as tf

# Force TensorFlow to use CPU.
# This avoids CUDA warnings/errors on machines without
# the required CUDA/cuDNN libraries.
tf.config.set_visible_devices([], "GPU")

from tensorflow.keras.models import load_model


# ============================================================
# PYTORCH
# ============================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is:
# Actual_APP/predict_weather.py
#
# Therefore PROJECT_ROOT is the folder containing this file.

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "Models"
TEMPERATURE_MODELS_DIR = PROJECT_ROOT / "Temperature_Models"
RAIN_MODELS_DIR = PROJECT_ROOT / "Rain_Models"
RAW_DATA_DIR = PROJECT_ROOT / "Raw Data"
RAW_MASTER_DATASET = RAW_DATA_DIR / "master_raw_weather.csv"

CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Faridabad": (28.4089, 77.3178),
    "Ghaziabad": (28.6692, 77.4538),
    "Gurugram": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
}

RAW_WEATHER_COLUMNS = [
    "datetime",
    "city",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "dew_point",
    "pressure",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "precipitation",
    "rain",
    "cloud_cover",
    "solar_radiation",
]


# ============================================================
# DATA FILES
# ============================================================

TEMPERATURE_DATASET = DATA_DIR / "temperature_dataset.csv"
RAIN_DATASET = DATA_DIR / "rain_dataset.csv"

# CNN files
CNN_LABELS = DATA_DIR / "rain_cnn_labels.csv"
CNN_IMAGE_DIR = DATA_DIR / "ncr_satellite_data"


# ============================================================
# TEMPERATURE MODEL FILES
# ============================================================

RIDGE_MODEL_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_ridge_model.pkl"
)

RIDGE_SCALER_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_scaler.pkl"
)

RIDGE_FEATURES_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_features.pkl"
)

XGB_MODEL_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_xgboost_model.pkl"
)

XGB_FEATURES_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_xgboost_features.pkl"
)

LSTM_MODEL_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_lstm_model.keras"
)

LSTM_SCALER_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_lstm_scaler.pkl"
)

LSTM_FEATURES_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_lstm_features.pkl"
)

LSTM_SEQUENCE_LENGTH_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_lstm_sequence_length.pkl"
)

TEMPERATURE_WEIGHTS_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_manual_weights.pkl"
)

TEMPERATURE_RANGE_PATH = (
    TEMPERATURE_MODELS_DIR / "temperature_prediction_range.pkl"
)


# ============================================================
# RAIN MODEL FILES
# ============================================================

RAIN_XGB_MODEL_PATH = (
    RAIN_MODELS_DIR / "rain_xgboost_model.pkl"
)

RAIN_XGB_FEATURES_PATH = (
    RAIN_MODELS_DIR / "rain_xgboost_features.pkl"
)

RAIN_RANGE_PATH = (
    RAIN_MODELS_DIR / "rain_prediction_range.pkl"
)

# CNN model
RAIN_CNN_MODEL_PATH = (
    RAIN_MODELS_DIR / "rain_cnn.pt"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_pickle(path):
    """Load a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)


def check_file(path, description):
    """Check whether a required file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n{path}"
        )


def get_feature_names(features):
    """
    Convert saved feature information into a normal list.

    Handles:
        list
        tuple
        numpy arrays
        pandas Index
        dictionaries
    """

    if isinstance(features, dict):
        # Some training scripts save feature metadata
        # inside a dictionary.
        for key in [
            "features",
            "feature_names",
            "columns",
            "selected_features"
        ]:
            if key in features:
                features = features[key]
                break

    return list(features)


def prepare_features(df, feature_names):
    """
    Create the exact feature dataframe expected by a model.

    Missing columns are filled with 0 so the prediction script
    does not crash if an unused feature is absent.
    """

    X = pd.DataFrame(index=df.index)

    for feature in feature_names:

        if feature in df.columns:
            X[feature] = df[feature]

        else:
            # Try numeric conversion if possible
            X[feature] = 0.0

    return X


def fetch_recent_weather(city):
    """Fetch recent hourly weather and persist it to the raw city datasets."""
    latitude, longitude = CITY_COORDINATES[city]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "precipitation",
                "rain",
                "cloud_cover",
                "shortwave_radiation",
            ],
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        },
        timeout=30,
    )
    response.raise_for_status()

    api_weather_df = pd.DataFrame(response.json()["hourly"])
    api_weather_df = api_weather_df.rename(columns={
        "time": "datetime",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dew_point",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "wind_gusts_10m": "wind_gust",
        "shortwave_radiation": "solar_radiation",
    })
    api_weather_df["city"] = city
    api_weather_df["latitude"] = latitude
    api_weather_df["longitude"] = longitude
    api_weather_df["datetime"] = pd.to_datetime(
        api_weather_df["datetime"]
    )

    latest_completed_hour = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .floor("h")
        .tz_localize(None)
    )
    api_weather_df = api_weather_df[
        api_weather_df["datetime"] <= latest_completed_hour
    ][RAW_WEATHER_COLUMNS].dropna(
        subset=["temperature", "humidity", "pressure", "rain"]
    )

    if api_weather_df.empty:
        raise ValueError(f"Weather API returned no usable data for {city}.")

    def append_rows(path, rows):
        if path.exists():
            existing = pd.read_csv(path)
            combined = pd.concat(
                [existing, rows],
                ignore_index=True,
            )
        else:
            combined = rows.copy()

        combined["datetime"] = pd.to_datetime(combined["datetime"])
        combined = combined.drop_duplicates(
            subset=["city", "datetime"],
            keep="last",
        ).sort_values(["city", "datetime"])
        combined.to_csv(path, index=False)

    append_rows(RAW_DATA_DIR / f"{city.lower()}.csv", api_weather_df)
    append_rows(RAW_MASTER_DATASET, api_weather_df)

    return api_weather_df.reset_index(drop=True)


def engineer_weather_features(raw_df):
    """Build the lag, rolling, time, and relationship features used in training."""
    df = raw_df.sort_values("datetime").copy().reset_index(drop=True)

    for column, lags in {
        "temperature": [1, 3, 6, 12, 24],
        "humidity": [1, 6, 24],
        "pressure": [1, 6, 24],
        "wind_speed": [1, 6, 24],
        "rain": [1, 6, 24],
    }.items():
        for lag in lags:
            df[f"{column}_lag_{lag}h"] = df[column].shift(lag)

    for column, periods in {
        "temperature": [1, 3, 6, 24],
        "humidity": [1, 6, 24],
        "pressure": [1, 3, 6, 24],
        "wind_speed": [1, 6],
    }.items():
        for period in periods:
            df[f"{column}_change_{period}h"] = (
                df[column] - df[column].shift(period)
            )

    for column, windows in {
        "temperature": [6, 12, 24],
        "humidity": [6, 24],
        "pressure": [6, 24],
        "wind_speed": [6, 24],
    }.items():
        for window in windows:
            df[f"{column}_mean_{window}h"] = (
                df[column].rolling(window).mean()
            )

    for column, windows in {
        "temperature": [6, 24],
        "humidity": [24],
        "pressure": [24],
    }.items():
        for window in windows:
            df[f"{column}_std_{window}h"] = (
                df[column].rolling(window).std()
            )

    for window in [6, 24, 72]:
        df[f"rain_sum_{window}h"] = df["rain"].rolling(window).sum()

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )
    df["day_of_year_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )
    df["temperature_dewpoint_difference"] = (
        df["temperature"] - df["dew_point"]
    )
    df["wind_gust_ratio"] = np.where(
        df["wind_speed"] != 0,
        df["wind_gust"] / df["wind_speed"],
        0,
    )
    df["pressure_change_rate"] = df["pressure"].diff()
    df["rain_intensity"] = df["rain"]

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


# ============================================================
# CNN ARCHITECTURE
# ============================================================

def create_cnn_model():
    """
    Recreate the exact ResNet18 architecture used by
    Models/cnn_model.py.
    """

    model = models.resnet18(weights=None)

    # The training script replaced the final layer with:
    #
    # nn.Linear(model.fc.in_features, 1)

    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    return model


# ============================================================
# CNN TRANSFORM
# ============================================================

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEMPERATURE MODELS
# ============================================================

print("\nLoading temperature models...")

# Check files
for path, description in [
    (RIDGE_MODEL_PATH, "Ridge model"),
    (RIDGE_SCALER_PATH, "Ridge scaler"),
    (RIDGE_FEATURES_PATH, "Ridge features"),
    (XGB_MODEL_PATH, "XGBoost model"),
    (XGB_FEATURES_PATH, "XGBoost features"),
    (LSTM_MODEL_PATH, "LSTM model"),
    (LSTM_SCALER_PATH, "LSTM scaler"),
    (LSTM_FEATURES_PATH, "LSTM features"),
    (LSTM_SEQUENCE_LENGTH_PATH, "LSTM sequence length"),
    (TEMPERATURE_WEIGHTS_PATH, "temperature ensemble weights"),
    (TEMPERATURE_RANGE_PATH, "temperature prediction range"),
]:
    check_file(path, description)


# Ridge
ridge_model = load_pickle(RIDGE_MODEL_PATH)
ridge_scaler = load_pickle(RIDGE_SCALER_PATH)
ridge_features = get_feature_names(
    load_pickle(RIDGE_FEATURES_PATH)
)


# XGBoost
xgb_temperature_model = load_pickle(
    XGB_MODEL_PATH
)

xgb_temperature_features = get_feature_names(
    load_pickle(XGB_FEATURES_PATH)
)


# LSTM
lstm_model = load_model(
    LSTM_MODEL_PATH,
    compile=False
)

lstm_scaler = load_pickle(
    LSTM_SCALER_PATH
)

lstm_features = get_feature_names(
    load_pickle(LSTM_FEATURES_PATH)
)

sequence_length = int(
    load_pickle(LSTM_SEQUENCE_LENGTH_PATH)
)


# Ensemble weights
temperature_weights = load_pickle(
    TEMPERATURE_WEIGHTS_PATH
)


# Temperature prediction range
temperature_range = load_pickle(
    TEMPERATURE_RANGE_PATH
)


print("Temperature models loaded!")
print("Temperature prediction range loaded!")


# ============================================================
# LOAD RAIN XGBOOST
# ============================================================

print("\nLoading rain model...")

check_file(
    RAIN_XGB_MODEL_PATH,
    "rain XGBoost model"
)

check_file(
    RAIN_XGB_FEATURES_PATH,
    "rain XGBoost features"
)

rain_xgb_model = load_pickle(
    RAIN_XGB_MODEL_PATH
)

rain_xgb_features = get_feature_names(
    load_pickle(RAIN_XGB_FEATURES_PATH)
)


# Rain range
if RAIN_RANGE_PATH.exists():

    rain_range = load_pickle(
        RAIN_RANGE_PATH
    )

    print("Rain prediction range loaded!")

else:

    rain_range = None

    print(
        "WARNING: Rain range file not found."
    )


# ============================================================
# LOAD CNN
# ============================================================

print("\nLoading CNN model...")

check_file(
    RAIN_CNN_MODEL_PATH,
    "CNN model"
)

cnn_device = torch.device("cpu")

cnn_model = create_cnn_model()

# Your rain_cnn.pt is a state_dict.
cnn_state_dict = torch.load(
    RAIN_CNN_MODEL_PATH,
    map_location=cnn_device,
    weights_only=True
)

cnn_model.load_state_dict(
    cnn_state_dict
)

cnn_model = cnn_model.to(cnn_device)

cnn_model.eval()

print("CNN model loaded!")
print("CNN device:", cnn_device)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nUsing Open-Meteo for recent weather data...")


# ============================================================
# AVAILABLE CITIES
# ============================================================

cities = sorted(CITY_COORDINATES)


print("\n==========================================")
print("AVAILABLE CITIES")
print("==========================================")

for i, city in enumerate(cities, 1):
    print(f"{i}. {city}")


# ============================================================
# CITY INPUT
# ============================================================

city_input = input(
    "\nEnter city name: "
).strip()

matching_cities = [
    city for city in cities
    if city.lower() == city_input.lower()
]

if not matching_cities:

    print(
        f"\nCity '{city_input}' not found."
    )

    print("\nAvailable cities:")

    for city in cities:
        print("-", city)

    raise SystemExit


city = matching_cities[0]


# ============================================================
# FETCH AND ENGINEER RECENT CITY DATA
# ============================================================

print(f"\nFetching recent weather for {city}...")
live_weather_df = fetch_recent_weather(city)
city_temperature_df = engineer_weather_features(live_weather_df)
city_rain_df = city_temperature_df

if city_temperature_df.empty:
    raise ValueError(
        f"Not enough usable recent weather data for {city} "
        "to calculate the model features."
    )


# ============================================================
# LATEST DATA
# ============================================================

latest_temperature_row = (
    city_temperature_df.iloc[-1]
)

latest_datetime = pd.to_datetime(
    latest_temperature_row["datetime"]
)

prediction_datetime = (
    latest_datetime + pd.Timedelta(days=1)
)


print("\n==========================================")
print("LATEST AVAILABLE DATA")
print("==========================================")

print(f"City: {city}")

print(
    f"Latest datetime: "
    f"{latest_datetime}"
)


# ============================================================
# PREPARE LATEST ROW
# ============================================================

latest_temperature_df = pd.DataFrame(
    [latest_temperature_row]
)


# ============================================================
# RIDGE PREDICTION
# ============================================================

ridge_X = prepare_features(
    latest_temperature_df,
    ridge_features
)

ridge_X_scaled = ridge_scaler.transform(
    ridge_X
)

ridge_prediction = float(
    ridge_model.predict(ridge_X_scaled)[0]
)


# ============================================================
# XGBOOST TEMPERATURE PREDICTION
# ============================================================

xgb_temperature_X = prepare_features(
    latest_temperature_df,
    xgb_temperature_features
)

xgb_temperature_prediction = float(
    xgb_temperature_model.predict(
        xgb_temperature_X
    )[0]
)


# ============================================================
# LSTM PREDICTION
# ============================================================

print(
    "\nPreparing LSTM sequence..."
)


if len(city_temperature_df) < sequence_length:

    raise ValueError(
        f"Not enough data for LSTM. "
        f"Required: {sequence_length}, "
        f"Available: {len(city_temperature_df)}"
    )


lstm_sequence_df = city_temperature_df.tail(
    sequence_length
).copy()


lstm_X = prepare_features(
    lstm_sequence_df,
    lstm_features
)


lstm_X_scaled = lstm_scaler.transform(
    lstm_X
)


lstm_X_sequence = np.array(
    lstm_X_scaled,
    dtype=np.float32
).reshape(
    1,
    sequence_length,
    len(lstm_features)
)


lstm_prediction = float(
    lstm_model.predict(
        lstm_X_sequence,
        verbose=0
    )[0][0]
)


# ============================================================
# TEMPERATURE ENSEMBLE
# ============================================================

def get_weight(weights, name):

    if isinstance(weights, dict):

        # Try normal names
        if name in weights:
            return float(weights[name])

        # Try alternative naming
        alternatives = {
            "Ridge": [
                "ridge",
                "ridge_weight"
            ],
            "XGBoost": [
                "xgboost",
                "xgb",
                "xgboost_weight"
            ],
            "LSTM": [
                "lstm",
                "lstm_weight"
            ]
        }

        for key in alternatives.get(name, []):

            if key in weights:
                return float(weights[key])

    elif isinstance(weights, (list, tuple, np.ndarray)):

        mapping = {
            "Ridge": 0,
            "XGBoost": 1,
            "LSTM": 2
        }

        return float(
            weights[mapping[name]]
        )

    raise ValueError(
        f"Could not find weight for {name}"
    )


ridge_weight = get_weight(
    temperature_weights,
    "Ridge"
)

xgb_weight = get_weight(
    temperature_weights,
    "XGBoost"
)

lstm_weight = get_weight(
    temperature_weights,
    "LSTM"
)


weight_total = (
    ridge_weight
    + xgb_weight
    + lstm_weight
)


# Normalize weights just in case
if weight_total <= 0:

    raise ValueError(
        "Temperature ensemble weights are invalid."
    )


ridge_weight /= weight_total
xgb_weight /= weight_total
lstm_weight /= weight_total


temperature_prediction = (
    ridge_prediction * ridge_weight
    + xgb_temperature_prediction * xgb_weight
    + lstm_prediction * lstm_weight
)


# ============================================================
# TEMPERATURE RANGE
# ============================================================

def extract_temperature_errors(range_data):

    """
    Extract lower/upper error bounds from the saved
    temperature range information.

    Supports both dictionary and tuple formats.
    """

    if isinstance(range_data, dict):

        # First try explicit names
        lower_keys = [
            "lower_error",
            "error_lower",
            "q025",
            "lower",
            "min_error"
        ]

        upper_keys = [
            "upper_error",
            "error_upper",
            "q975",
            "upper",
            "max_error"
        ]

        lower = None
        upper = None

        for key in lower_keys:
            if key in range_data:
                lower = float(range_data[key])
                break

        for key in upper_keys:
            if key in range_data:
                upper = float(range_data[key])
                break

        if lower is not None and upper is not None:
            return lower, upper

        # Some versions save quantiles
        if (
            "error_2.5%" in range_data
            and "error_97.5%" in range_data
        ):
            return (
                float(range_data["error_2.5%"]),
                float(range_data["error_97.5%"])
            )

        # Search nested dictionaries
        for value in range_data.values():

            if isinstance(value, dict):

                try:
                    return extract_temperature_errors(
                        value
                    )
                except Exception:
                    pass

    raise ValueError(
        "Could not determine temperature "
        "prediction error range."
    )


try:

    temp_error_low, temp_error_high = (
        extract_temperature_errors(
            temperature_range
        )
    )

    temperature_lower = max(
        -100,
        temperature_prediction + temp_error_low
    )

    temperature_upper = min(
        100,
        temperature_prediction + temp_error_high
    )

except Exception as e:

    print(
        f"\nWARNING: Could not load temperature "
        f"range correctly: {e}"
    )

    temperature_lower = temperature_prediction
    temperature_upper = temperature_prediction


# ============================================================
# RAIN XGBOOST PREDICTION
# ============================================================

rain_latest_row = city_rain_df.iloc[-1]

rain_latest_df = pd.DataFrame(
    [rain_latest_row]
)


rain_X = prepare_features(
    rain_latest_df,
    rain_xgb_features
)


rain_prediction = float(
    rain_xgb_model.predict(
        rain_X
    )[0]
)


# Rainfall cannot physically be negative
rain_prediction = max(
    0.0,
    rain_prediction
)


# ============================================================
# RAIN RANGE
# ============================================================

if rain_range is not None:

    try:

        if isinstance(rain_range, dict):

            lower_error = None
            upper_error = None

            # Current tighter-range format
            for key in [
                "lower_error",
                "error_lower",
                "q25",
                "25% error",
                "error_25"
            ]:

                if key in rain_range:
                    lower_error = float(
                        rain_range[key]
                    )
                    break

            for key in [
                "upper_error",
                "error_upper",
                "q75",
                "75% error",
                "error_75"
            ]:

                if key in rain_range:
                    upper_error = float(
                        rain_range[key]
                    )
                    break

            # Older format fallback
            if lower_error is None:

                for key in [
                    "error_2.5%",
                    "2.5% error",
                    "q025"
                ]:

                    if key in rain_range:
                        lower_error = float(
                            rain_range[key]
                        )
                        break

            if upper_error is None:

                for key in [
                    "error_97.5%",
                    "97.5% error",
                    "q975"
                ]:

                    if key in rain_range:
                        upper_error = float(
                            rain_range[key]
                        )
                        break

            if (
                lower_error is None
                or upper_error is None
            ):

                raise ValueError(
                    "Rain range keys not found."
                )

        else:

            raise ValueError(
                "Unexpected rain range format."
            )


        rain_lower = max(
            0.0,
            rain_prediction + lower_error
        )

        rain_upper = max(
            rain_lower,
            rain_prediction + upper_error
        )

        # Small numerical cleanup
        rain_lower = round(
            rain_lower,
            2
        )

        rain_upper = round(
            rain_upper,
            2
        )

    except Exception as e:

        print(
            f"\nWARNING: Could not use rain "
            f"prediction range: {e}"
        )

        rain_lower = rain_prediction
        rain_upper = rain_prediction

else:

    rain_lower = rain_prediction
    rain_upper = rain_prediction


# ============================================================
# RAIN CATEGORY
# ============================================================

def classify_rain(mm):

    if mm <= 0.0:
        return "No rain"

    elif mm < 0.1:
        return "Very light"

    elif mm < 2.5:
        return "Light"

    elif mm < 7.6:
        return "Moderate"

    elif mm < 15.0:
        return "Heavy"

    elif mm < 30.0:
        return "Very heavy"

    else:
        return "Extreme"


rain_category = classify_rain(
    rain_prediction
)


# ============================================================
# CNN SATELLITE PREDICTION
# ============================================================

cnn_probability = None
cnn_prediction = None
cnn_image_date = None
cnn_image_path = None


# Find satellite image corresponding to prediction date
date_string = prediction_datetime.strftime(
    "%Y-%m-%d"
)

expected_image = (
    CNN_IMAGE_DIR
    / f"ncr_{date_string}.png"
)


if expected_image.exists():

    cnn_image_path = expected_image
    cnn_image_date = date_string

else:

    # If exact next-day image is unavailable,
    # find the newest image available before the
    # prediction date.

    image_files = sorted(
        CNN_IMAGE_DIR.glob("ncr_*.png")
    )

    valid_images = []

    for image_file in image_files:

        try:

            image_date = pd.to_datetime(
                image_file.stem.replace(
                    "ncr_",
                    ""
                )
            )

            if image_date <= prediction_datetime:
                valid_images.append(
                    (image_date, image_file)
                )

        except Exception:
            continue

    if valid_images:

        valid_images.sort(
            key=lambda x: x[0]
        )

        cnn_image_date, cnn_image_path = (
            valid_images[-1]
        )


# Run CNN if an image exists
if cnn_image_path is not None:

    try:

        img = Image.open(
            cnn_image_path
        ).convert("RGBA")

        # Same preprocessing used during training:
        #
        # RGBA -> white RGB background
        # -> resize
        # -> tensor
        # -> ImageNet normalization

        bg = Image.new(
            "RGB",
            img.size,
            (255, 255, 255)
        )

        bg.paste(
            img,
            mask=img.split()[3]
        )

        img_tensor = cnn_transform(
            bg
        ).unsqueeze(0).to(
            cnn_device
        )


        with torch.no_grad():

            logits = cnn_model(
                img_tensor
            )

            cnn_probability = float(
                torch.sigmoid(
                    logits
                ).item()
            )


        cnn_prediction = (
            "Rain"
            if cnn_probability >= 0.5
            else "No rain"
        )

    except Exception as e:

        print(
            f"\nWARNING: CNN prediction failed: {e}"
        )

else:

    print(
        "\nWARNING: No suitable satellite "
        "image found for CNN prediction."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("NEXT-DAY WEATHER PREDICTION")
print("==========================================")

print(
    f"City              : {city}"
)

print(
    f"Latest data       : {latest_datetime}"
)

print(
    f"Prediction time   : {prediction_datetime}"
)

print("------------------------------------------")

print(
    f"Temperature       : "
    f"{temperature_prediction:.2f} °C"
)

print(
    f"Temperature range : "
    f"{temperature_lower:.2f} - "
    f"{temperature_upper:.2f} °C"
)

print()

print(
    f"Rainfall          : "
    f"{rain_prediction:.2f} mm"
)

print(
    f"Rainfall range    : "
    f"{rain_lower:.2f} - "
    f"{rain_upper:.2f} mm"
)

print(
    f"Rain category     : "
    f"{rain_category}"
)


# ============================================================
# CNN OUTPUT
# ============================================================

print("\n------------------------------------------")
print("SATELLITE CNN")
print("------------------------------------------")

if cnn_prediction is not None:

    print(
        f"Satellite image  : "
        f"{cnn_image_date}"
    )

    print(
        f"CNN prediction   : "
        f"{cnn_prediction}"
    )

    print(
        f"Rain probability : "
        f"{cnn_probability * 100:.2f}%"
    )

else:

    print(
        "CNN prediction   : Not available"
    )

    print(
        "Reason            : "
        "No suitable satellite image found"
    )


# ============================================================
# TEMPERATURE MODEL DETAILS
# ============================================================

print("\nTemperature model predictions:")

print(
    f"Ridge             : "
    f"{ridge_prediction:.2f} °C"
)

print(
    f"XGBoost           : "
    f"{xgb_temperature_prediction:.2f} °C"
)

print(
    f"LSTM              : "
    f"{lstm_prediction:.2f} °C"
)


print("\nTemperature ensemble weights:")

print(
    f"Ridge             : "
    f"{ridge_weight:.6f}"
)

print(
    f"XGBoost           : "
    f"{xgb_weight:.6f}"
)

print(
    f"LSTM              : "
    f"{lstm_weight:.6f}"
)

print(
    f"Total             : "
    f"{(
        ridge_weight
        + xgb_weight
        + lstm_weight
    ):.6f}"
)


# ============================================================
# RAIN MODEL DETAILS
# ============================================================

print("\nRain model:")
print("XGBoost")


# ============================================================
# FINISHED
# ============================================================

print("\n==========================================")
print("PREDICTION COMPLETED")
print("==========================================")