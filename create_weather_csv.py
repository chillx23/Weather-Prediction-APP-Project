from datetime import datetime, timedelta

import requests
import pandas as pd
import os


# ============================================================
# 1. CITY INFORMATION
# ============================================================

cities = {
    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090
    },

    "Noida": {
        "latitude": 28.5355,
        "longitude": 77.3910
    },

    "Ghaziabad": {
        "latitude": 28.6692,
        "longitude": 77.4538
    },

    "Gurugram": {
        "latitude": 28.4595,
        "longitude": 77.0266
    },

    "Faridabad": {
        "latitude": 28.4089,
        "longitude": 77.3178
    }
}


# ============================================================
# 2. DATE RANGE
# ============================================================

START_DATE = "2022-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


# ============================================================
# 3. OUTPUT FOLDER
# ============================================================

output_folder = "data"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 4. FUNCTION TO DOWNLOAD WEATHER DATA
# ============================================================

def get_weather_data(city_name, latitude, longitude):

    print(f"\nDownloading data for {city_name}...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "start_date": START_DATE,
        "end_date": END_DATE,

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
            "shortwave_radiation"
        ],

        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error:", response.status_code)
        print(response.text)
        return None

    data = response.json()

    # --------------------------------------------------------
    # Convert API response into DataFrame
    # --------------------------------------------------------

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    # --------------------------------------------------------
    # Rename columns according to our project
    # --------------------------------------------------------

    df.rename(columns={
        "time": "datetime",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dew_point",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "wind_gusts_10m": "wind_gust",
        "shortwave_radiation": "solar_radiation"
    }, inplace=True)

    # --------------------------------------------------------
    # Add city information
    # --------------------------------------------------------

    df["city"] = city_name
    df["latitude"] = latitude
    df["longitude"] = longitude

    # --------------------------------------------------------
    # Arrange columns
    # --------------------------------------------------------

    columns = [
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
        "solar_radiation"
    ]

    df = df[columns]

    return df


# ============================================================
# 5. DOWNLOAD DATA FOR ALL FIVE CITIES
# ============================================================

all_data = []


for city, location in cities.items():

    df = get_weather_data(
        city,
        location["latitude"],
        location["longitude"]
    )

    if df is not None:

        # ----------------------------------------------------
        # Save individual city CSV
        # ----------------------------------------------------

        filename = city.lower() + ".csv"

        filepath = os.path.join(
            output_folder,
            filename
        )

        df.to_csv(
            filepath,
            index=False
        )

        print(f"Saved: {filepath}")
        print(f"Rows: {len(df)}")

        # Add to master dataset
        all_data.append(df)


# ============================================================
# 6. COMBINE ALL FIVE DATASETS
# ============================================================

if len(all_data) > 0:

    master_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Sort by city and datetime
    master_df.sort_values(
        ["city", "datetime"],
        inplace=True
    )

    # Save master CSV
    master_path = os.path.join(
        output_folder,
        "master_raw_weather.csv"
    )

    master_df.to_csv(
        master_path,
        index=False
    )

    print("\n====================================")
    print("DATASET CREATION COMPLETE")
    print("====================================")

    print(f"Master dataset: {master_path}")
    print(f"Total rows: {len(master_df)}")
    print(f"Total columns: {len(master_df.columns)}")

    print("\nCities:")
    print(master_df["city"].value_counts())

    print("\nColumns:")
    print(list(master_df.columns))

else:

    print("No data was downloaded.")