import pandas as pd
import os


# ============================================================
# 1. LOAD FEATURE DATASET
# ============================================================

input_file = "data/feature_dataset.csv"

df = pd.read_csv(input_file)

print("Feature dataset loaded!")
print("Shape:", df.shape)


# ============================================================
# 2. CONVERT DATETIME
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])


# ============================================================
# 3. SORT DATA
# ============================================================

df = df.sort_values(
    ["city", "datetime"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE NEXT-DAY TARGETS
# ============================================================
#
# Dataset is hourly.
#
# shift(-24) means:
#
# Current hour  ---> Same hour tomorrow
#
# Example:
#
# 2025-01-01 10:00 ---> 2025-01-02 10:00
#
# 2025-01-01 11:00 ---> 2025-01-02 11:00
#
# ============================================================

df["temperature_target_next_day"] = (
    df.groupby("city")["temperature"].shift(-24)
)

df["rain_target_next_day"] = (
    df.groupby("city")["rain"].shift(-24)
)


# ============================================================
# 5. REMOVE ROWS WITHOUT NEXT-DAY TARGET
# ============================================================

df = df.dropna(
    subset=[
        "temperature_target_next_day",
        "rain_target_next_day"
    ]
).reset_index(drop=True)


# ============================================================
# 6. TEMPERATURE DATASET
# ============================================================

temperature_df = df.copy()

temperature_df.drop(
    columns=["rain_target_next_day"],
    inplace=True
)


# ============================================================
# 7. RAIN DATASET
# ============================================================

rain_df = df.copy()

rain_df.drop(
    columns=["temperature_target_next_day"],
    inplace=True
)


# ============================================================
# 8. SAVE DATASETS
# ============================================================

temperature_file = "data/temperature_dataset.csv"

rain_file = "data/rain_dataset.csv"


temperature_df.to_csv(
    temperature_file,
    index=False
)

rain_df.to_csv(
    rain_file,
    index=False
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("NEXT-DAY DATASETS CREATED")
print("==========================================")

print("\nTemperature Dataset")
print("------------------------------------------")
print("File:", temperature_file)
print("Rows:", len(temperature_df))
print("Columns:", len(temperature_df.columns))
print("Target: temperature_target_next_day")


print("\nRain Dataset")
print("------------------------------------------")
print("File:", rain_file)
print("Rows:", len(rain_df))
print("Columns:", len(rain_df.columns))
print("Target: rain_target_next_day")


# ============================================================
# 10. SHOW EXAMPLE
# ============================================================

print("\n==========================================")
print("EXAMPLE NEXT-DAY TARGET")
print("==========================================")

example = temperature_df[
    [
        "city",
        "datetime",
        "temperature",
        "temperature_target_next_day"
    ]
].head(10)

print(example)