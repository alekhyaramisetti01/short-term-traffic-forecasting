# src/preprocess_noaa.py
import os, argparse
from pyspark.sql import SparkSession, functions as F

NYC_TZ = "America/New_York"
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    ap = argparse.ArgumentParser(description="NOAA → 15-minute weather with optional cleanup")
    ap.add_argument("--infile", default=os.path.join(BASE, "data", "raw","noaa", "72505394728.csv"),
                    help="Path to NOAA CSV (default: data/raw/noaa/72505394728.csv)")
    ap.add_argument("--outdir", default=os.path.join(BASE, "data", "processed", "noaa", "weather_15m"),
                    help="Output Parquet dir (default: data/processed/noaa/weather_15m)")
    ap.add_argument("--start", default="2025-01-01 00:00:00", help="Start (NYC local)")
    ap.add_argument("--end",   default="2025-08-31 23:59:59", help="End (NYC local)")
    ap.add_argument("--no-clean", action="store_true", help="Disable cleanup (keep raw agg values)")
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("NOAA_Weather_Preprocessing")
        .config("spark.sql.session.timeZone", NYC_TZ)
        .getOrCreate()
    )

    # 1) Read CSV
    df = spark.read.option("header", True).csv(args.infile)

    # 2) Parse timestamps (NOAA DATE is UTC)
    df = df.withColumn("datetime_utc", F.to_timestamp("DATE"))

    # 3) Robust numeric parsing
    cols = df.columns

    # Temperature / Dew: tenths °C in first subfield "0123,QC"
    temp = (F.split(F.col("TMP"), ",")[0].cast("float") / 10.0) if "TMP" in cols else F.lit(None).cast("float")
    dew  = (F.split(F.col("DEW"), ",")[0].cast("float") / 10.0) if "DEW" in cols else F.lit(None).cast("float")

    # Wind speed: WND = ddd,ssss,qq,ff (ssss in tenths m/s)
    wind = (F.split(F.col("WND"), ",")[1].cast("float") / 10.0) if "WND" in cols else F.lit(None).cast("float")

    # Precip: prefer PRCP first field (tenths mm), else AA1 first field (mm)
    precip = None
    if "PRCP" in cols:
        precip = (F.split(F.col("PRCP"), ",")[0].cast("float") / 10.0)
    elif "AA1" in cols:
        precip = (F.split(F.col("AA1"), ",")[0].cast("float"))
    else:
        precip = F.lit(None).cast("float")

    df = (df.withColumn("temperature_C", temp)
            .withColumn("dew_point_C", dew)
            .withColumn("wind_speed_mps", wind)
            .withColumn("precip_mm", precip))

    # 4) UTC → NYC, clamp to window in local time
    df = df.withColumn("datetime_local", F.from_utc_timestamp("datetime_utc", NYC_TZ))
    df = df.where((F.col("datetime_local") >= F.lit(args.start)) &
                  (F.col("datetime_local") <= F.lit(args.end)))

    # 5) Floor to 15-minute bins
    df = df.withColumn(
        "bin_15m",
        F.from_unixtime((F.unix_timestamp("datetime_local")/900).cast("bigint")*900).cast("timestamp")
    )

    # 6) Aggregate to 15-min (mean for continuous, sum for precip)
    agg = (df.groupBy("bin_15m")
             .agg(F.avg("temperature_C").alias("temp_C"),
                  F.avg("dew_point_C").alias("dew_C"),
                  F.avg("wind_speed_mps").alias("wind_mps"),
                  F.sum("precip_mm").alias("precip_mm"))
             .orderBy("bin_15m"))

    # 7) Optional cleanup (recommended)
    if not args.no_clean:
        # Remove NOAA sentinel spikes like 999.9/669.2 by nulling absurd temps/dew
        agg = (agg.withColumn("temp_C", F.when((F.col("temp_C") > 60) | (F.col("temp_C") < -60), None).otherwise(F.col("temp_C")))
                   .withColumn("dew_C",  F.when((F.col("dew_C")  > 60) | (F.col("dew_C")  < -60), None).otherwise(F.col("dew_C")))
                   # Wind/precip cannot be negative; clip & fill precip nulls with 0
                   .withColumn("wind_mps",  F.when(F.col("wind_mps") < 0, 0).otherwise(F.col("wind_mps")))
                   .withColumn("precip_mm", F.when(F.col("precip_mm") < 0, 0).otherwise(F.col("precip_mm"))))
        agg = agg.fillna({"precip_mm": 0.0})

    # 8) Write parquet
    agg.write.mode("overwrite").parquet(args.outdir)
    print(f"[OK] NOAA 15-min saved → {args.outdir}")
    spark.stop()

if __name__ == "__main__":
    main()
