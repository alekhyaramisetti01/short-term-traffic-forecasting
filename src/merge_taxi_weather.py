# src/merge_taxi_weather.py
import os
from pyspark.sql import SparkSession, functions as F

NYC_TZ = "America/New_York"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAXI_DIR = os.path.join(BASE, "data", "processed", "nyc_taxi", "taxi_demand_15m")   # fixed
WEATHER_DIR = os.path.join(BASE, "data", "processed", "noaa", "weather_15m")
OUT_DIR = os.path.join(BASE, "data", "processed", "merged")
OUT_FILE = os.path.join(OUT_DIR, "taxi_weather_15m")

def main():
    spark = (
        SparkSession.builder
        .appName("Merge_Taxi_Weather")
        .config("spark.sql.session.timeZone", NYC_TZ)
        .getOrCreate()
    )

    # Read
    taxi = spark.read.parquet(TAXI_DIR).dropDuplicates(["bin_15m"])
    weather = spark.read.parquet(WEATHER_DIR).dropDuplicates(["bin_15m"])

    # Left join on 15-min bins
    merged = taxi.join(weather, on="bin_15m", how="left").orderBy("bin_15m")

    # Fill missing weather (optional)
    merged = merged.fillna({"temp_C": 0.0, "dew_C": 0.0, "wind_mps": 0.0, "precip_mm": 0.0})

    # Write
    os.makedirs(OUT_DIR, exist_ok=True)          # fixed
    merged.write.mode("overwrite").parquet(OUT_FILE)

    # Small CSV sample
    (merged.orderBy(F.col("bin_15m").desc()).limit(200)
           .coalesce(1)
           .write.mode("overwrite").option("header", True)
           .csv(os.path.join(OUT_DIR, "sample_merged_csv")))

    print(f"[OK] Merged taxi + weather dataset saved → {OUT_FILE}")
    spark.stop()

if __name__ == "__main__":
    main()
