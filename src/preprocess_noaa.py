import os
from pyspark.sql import SparkSession, functions as F

NYC_TZ = "America/New_York"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_FILE = os.path.join(BASE, "data", "raw","noaa", "72505394728.csv")
OUT_DIR = os.path.join(BASE, "data", "processed", "noaa")
OUT_FILE = os.path.join(OUT_DIR, "weather_15m")

def col_exists(df, name):  # convenience helper
    return name in df.columns

def main():
    spark = (
        SparkSession.builder
        .appName("NOAA_Weather_Preprocessing")
        .config("spark.sql.session.timeZone", NYC_TZ)
        .getOrCreate()
    )

    # 1) Read raw NOAA CSV (header present; let Spark infer)
    df = spark.read.option("header", True).csv(IN_FILE)

    # 2) Parse timestamps (NOAA DATE is UTC)
    df = df.withColumn("datetime_utc", F.to_timestamp("DATE"))

    # 3) Build robust numeric columns (fields may vary by year)
    # Temperature & dew point: NOAA encodes like "0123,1" (tenths °C, qc)
    if col_exists(df, "TMP"):
        df = df.withColumn("temperature_C", (F.split(F.col("TMP"), ",")[0].cast("float") / 10.0))
    else:
        df = df.withColumn("temperature_C", F.lit(None).cast("float"))

    if col_exists(df, "DEW"):
        df = df.withColumn("dew_point_C", (F.split(F.col("DEW"), ",")[0].cast("float") / 10.0))
    else:
        df = df.withColumn("dew_point_C", F.lit(None).cast("float"))

    # Wind speed: WND field like "ddd,ssss,qq,ff" where ssss = tenths of m/s
    if col_exists(df, "WND"):
        df = df.withColumn("wind_speed_mps", (F.split(F.col("WND"), ",")[1].cast("float") / 10.0))
    else:
        df = df.withColumn("wind_speed_mps", F.lit(None).cast("float"))

    # Precipitation:
    # Prefer PRCP if present (first subfield often tenths of mm)
    # Otherwise AA1 (last-hour precip). AA1 structure: aaaa,pppp,xx,qq
    # We'll take the first numeric part; units already mm in AA1 (no /10).
    if col_exists(df, "PRCP"):
        precip_mm = (F.split(F.col("PRCP"), ",")[0].cast("float") / 10.0)
    elif col_exists(df, "AA1"):
        precip_mm = F.split(F.col("AA1"), ",")[0].cast("float")
    else:
        precip_mm = F.lit(0.0)

    df = df.withColumn("precip_mm", precip_mm)

    # 4) Convert to NYC local time & floor to 15-min bins
    df = df.withColumn("datetime_local", F.from_utc_timestamp("datetime_utc", NYC_TZ))
    df = df.withColumn(
        "bin_15m",
        F.from_unixtime((F.unix_timestamp("datetime_local")/900).cast("bigint")*900).cast("timestamp")
    )

    # 5) Aggregate to 15-min
    agg = (
        df.groupBy("bin_15m")
          .agg(
              F.avg("temperature_C").alias("temp_C"),
              F.avg("dew_point_C").alias("dew_C"),
              F.avg("wind_speed_mps").alias("wind_mps"),
              F.sum("precip_mm").alias("precip_mm")
          )
          .orderBy("bin_15m")
    )

    os.makedirs(OUT_FILE, exist_ok=True)
    agg.write.mode("overwrite").parquet(OUT_FILE)
    print(f"[OK] Processed NOAA weather → {OUT_FILE}")

    spark.stop()

if __name__ == "__main__":
    main()
