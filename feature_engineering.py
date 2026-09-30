import pandas as pd
import numpy as np
import os


# ============================================================
# 1. LOAD MASTER RAW DATASET
# ============================================================

input_file = "data/master_raw_weather.csv"

df = pd.read_csv(input_file)

print("Raw dataset loaded!")
print("Shape:", df.shape)


# ============================================================
# 2. CONVERT DATETIME
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

# Sort by city and time
df = df.sort_values(
    ["city", "datetime"]
).reset_index(drop=True)


# ============================================================
# 3. CREATE LAG FEATURES
# ============================================================

print("\nCreating lag features...")


# ----------------------------
# Temperature
# ----------------------------

for lag in [1, 3, 6, 12, 24]:
    df[f"temperature_lag_{lag}h"] = (
        df.groupby("city")["temperature"]
        .shift(lag)
    )


# ----------------------------
# Humidity
# ----------------------------

for lag in [1, 6, 24]:
    df[f"humidity_lag_{lag}h"] = (
        df.groupby("city")["humidity"]
        .shift(lag)
    )


# ----------------------------
# Pressure
# ----------------------------

for lag in [1, 6, 24]:
    df[f"pressure_lag_{lag}h"] = (
        df.groupby("city")["pressure"]
        .shift(lag)
    )


# ----------------------------
# Wind Speed
# ----------------------------

for lag in [1, 6, 24]:
    df[f"wind_speed_lag_{lag}h"] = (
        df.groupby("city")["wind_speed"]
        .shift(lag)
    )


# ----------------------------
# Rain
# ----------------------------

for lag in [1, 6, 24]:
    df[f"rain_lag_{lag}h"] = (
        df.groupby("city")["rain"]
        .shift(lag)
    )


# ============================================================
# 4. CHANGE / TREND FEATURES
# ============================================================

print("Creating change/trend features...")


# ----------------------------
# Temperature Change
# ----------------------------

for period in [1, 3, 6, 24]:
    df[f"temperature_change_{period}h"] = (
        df["temperature"]
        - df.groupby("city")["temperature"].shift(period)
    )


# ----------------------------
# Humidity Change
# ----------------------------

for period in [1, 6, 24]:
    df[f"humidity_change_{period}h"] = (
        df["humidity"]
        - df.groupby("city")["humidity"].shift(period)
    )


# ----------------------------
# Pressure Change
# ----------------------------

for period in [1, 3, 6, 24]:
    df[f"pressure_change_{period}h"] = (
        df["pressure"]
        - df.groupby("city")["pressure"].shift(period)
    )


# ----------------------------
# Wind Speed Change
# ----------------------------

for period in [1, 6]:
    df[f"wind_speed_change_{period}h"] = (
        df["wind_speed"]
        - df.groupby("city")["wind_speed"].shift(period)
    )


# ============================================================
# 5. ROLLING FEATURES
# ============================================================

print("Creating rolling features...")


# ----------------------------
# Temperature
# ----------------------------

for window in [6, 12, 24]:

    df[f"temperature_mean_{window}h"] = (
        df.groupby("city")["temperature"]
        .transform(
            lambda x: x.rolling(window).mean()
        )
    )


for window in [6, 24]:

    df[f"temperature_std_{window}h"] = (
        df.groupby("city")["temperature"]
        .transform(
            lambda x: x.rolling(window).std()
        )
    )


# ----------------------------
# Humidity
# ----------------------------

for window in [6, 24]:

    df[f"humidity_mean_{window}h"] = (
        df.groupby("city")["humidity"]
        .transform(
            lambda x: x.rolling(window).mean()
        )
    )


df["humidity_std_24h"] = (
    df.groupby("city")["humidity"]
    .transform(
        lambda x: x.rolling(24).std()
    )
)


# ----------------------------
# Pressure
# ----------------------------

for window in [6, 24]:

    df[f"pressure_mean_{window}h"] = (
        df.groupby("city")["pressure"]
        .transform(
            lambda x: x.rolling(window).mean()
        )
    )


df["pressure_std_24h"] = (
    df.groupby("city")["pressure"]
    .transform(
        lambda x: x.rolling(24).std()
    )
)


# ----------------------------
# Wind Speed
# ----------------------------

for window in [6, 24]:

    df[f"wind_speed_mean_{window}h"] = (
        df.groupby("city")["wind_speed"]
        .transform(
            lambda x: x.rolling(window).mean()
        )
    )


# ----------------------------
# Rain
# ----------------------------

for window in [6, 24, 72]:

    df[f"rain_sum_{window}h"] = (
        df.groupby("city")["rain"]
        .transform(
            lambda x: x.rolling(window).sum()
        )
    )


# ============================================================
# 6. TIME FEATURES
# ============================================================

print("Creating time features...")


df["hour"] = df["datetime"].dt.hour

df["day_of_week"] = df["datetime"].dt.dayofweek

df["day_of_year"] = df["datetime"].dt.dayofyear

df["month"] = df["datetime"].dt.month


# ----------------------------
# Cyclic Features
# ----------------------------

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)


df["day_of_year_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["day_of_year_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


# ============================================================
# 7. PHYSICS / WEATHER RELATIONSHIP FEATURES
# ============================================================

print("Creating physics/weather relationship features...")


# ----------------------------
# Temperature - Dew Point
# ----------------------------

df["temperature_dewpoint_difference"] = (
    df["temperature"] - df["dew_point"]
)


# ----------------------------
# Wind Gust Ratio
# ----------------------------

df["wind_gust_ratio"] = np.where(
    df["wind_speed"] != 0,
    df["wind_gust"] / df["wind_speed"],
    0
)


# ----------------------------
# Pressure Change Rate
# ----------------------------

df["pressure_change_rate"] = (
    df["pressure"]
    - df.groupby("city")["pressure"].shift(1)
)


# ----------------------------
# Rain Intensity
# ----------------------------

df["rain_intensity"] = df["rain"]


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")


# Lag and rolling features naturally create NaN values
# at the beginning of each city's dataset.

df = df.dropna().reset_index(drop=True)


# ============================================================
# 9. SAVE FEATURE DATASET
# ============================================================

output_file = "data/feature_dataset.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n============================================")
print("FEATURE ENGINEERING COMPLETE")
print("============================================")

print("Output:", output_file)

print("Rows:", len(df))

print("Columns:", len(df.columns))

print("\nCities:")
print(df["city"].value_counts())

print("\nFeature Dataset Columns:")
for column in df.columns:
    print(column)