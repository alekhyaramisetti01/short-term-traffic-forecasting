# src/preprocess_taxi.py
import os, glob
from pyspark.sql import SparkSession, functions as F

NYC_TZ = "America/New_York"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(BASE, "data", "raw","nyc_taxi")
OUT_DIR = os.path.join(BASE, "data", "processed", "nyc")
OUT_TAXI_15M = os.path.join(OUT_DIR, "taxi_demand_15m")

def list_parquets(path):
    paths = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not paths:
        raise SystemExit(f"[ERROR] No parquet files found in {path}")
    return paths

def floor_to_15min(col_ts):
    # Floor timestamp to 15-minute interval
    return F.from_unixtime((F.unix_timestamp(col_ts)/900).cast("bigint")*900).cast("timestamp")

def main():
    spark = (
        SparkSession.builder
        .appName("Taxi_15m_Aggregation")
        .config("spark.sql.session.timeZone", NYC_TZ)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    # 1. Read all Parquet files
    paths = list_parquets(IN_DIR)
    df = spark.read.parquet(*paths).select(
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID"
    )

    # 2. Basic cleaning
    df = (
        df.where(F.col("tpep_pickup_datetime").isNotNull())
          .where(F.col("tpep_dropoff_datetime").isNotNull())
          .where(F.col("PULocationID").isNotNull())
          .where(
              (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime")) &
              (F.col("tpep_dropoff_datetime") <= F.col("tpep_pickup_datetime") + F.expr("INTERVAL 24 HOURS"))
          )
    )

    # 3. Add 15-minute time bin
    df = df.withColumn("bin_15m", floor_to_15min(F.col("tpep_pickup_datetime")))

    # 4. Aggregate
    agg = (
        df.groupBy("bin_15m")
          .agg(
              F.count(F.lit(1)).alias("trips_15m"),
              F.countDistinct("PULocationID").alias("unique_PUs_15m")
          )
    )

    # 5. Build continuous 15-min bins
    bounds = agg.agg(F.min("bin_15m").alias("min_ts"), F.max("bin_15m").alias("max_ts")).collect()[0]
    start, end = bounds["min_ts"], bounds["max_ts"]
    if start is None or end is None:
        raise SystemExit("[ERROR] No rows after filtering; check your input data.")

    cal = (
        spark.createDataFrame([(start, end)], ["start_ts", "end_ts"])
             .select(F.explode(F.sequence(F.col("start_ts"), F.col("end_ts"), F.expr("interval 15 minutes"))).alias("bin_15m"))
    )

    full = (
        cal.join(agg, on="bin_15m", how="left")
           .select(
               "bin_15m",
               F.coalesce(F.col("trips_15m"), F.lit(0)).cast("long").alias("trips_15m"),
               F.coalesce(F.col("unique_PUs_15m"), F.lit(0)).cast("long").alias("unique_PUs_15m")
           )
           .orderBy("bin_15m")
    )

    # 6. Save results
    os.makedirs(OUT_TAXI_15M, exist_ok=True)

    full.write.mode("overwrite").parquet(OUT_TAXI_15M)

    # Small CSV sample (last 200 rows)
    full.orderBy(F.col("bin_15m").desc()).limit(200).coalesce(1).write.mode("overwrite") \
        .option("header", True) \
        .csv(os.path.join(OUT_DIR, "sample_taxi_15m_csv"))

    print(f"[OK] Processed files stored at: {OUT_TAXI_15M}")
    spark.stop()

if __name__ == "__main__":
    main()
