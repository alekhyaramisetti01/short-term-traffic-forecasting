import os, glob, argparse
from pyspark.sql import SparkSession, functions as F

NYC_TZ = "America/New_York"
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(BASE, "data","raw", "nyc_taxi")
OUT_DIR = os.path.join(BASE, "data", "processed", "nyc_taxi")
OUT_TAXI_15M = os.path.join(OUT_DIR, "taxi_demand_15m")

def list_parquets(path):
    paths = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not paths:
        raise SystemExit(f"[ERROR] No parquet files found in {path}")
    return paths

def floor_to_15min(col_ts):
    # floor timestamp to 15-min (900 sec)
    return F.from_unixtime((F.unix_timestamp(col_ts)/900).cast("bigint")*900).cast("timestamp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01 00:00:00")
    ap.add_argument("--end",   default="2025-08-31 23:59:59")
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("Taxi_15m_Aggregation")
        .config("spark.sql.session.timeZone", NYC_TZ)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    start_lit = F.lit(args.start).cast("timestamp")
    end_lit   = F.lit(args.end).cast("timestamp")

    paths = list_parquets(IN_DIR)
    df = spark.read.parquet(*paths).select(
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID"
    )

    # hygiene + sane durations + hard clamp to window
    df = (
        df.where(F.col("tpep_pickup_datetime").isNotNull())
          .where(F.col("tpep_dropoff_datetime").isNotNull())
          .where(F.col("PULocationID").isNotNull())
          .where(
              (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime")) &
              (F.col("tpep_dropoff_datetime") <= F.col("tpep_pickup_datetime") + F.expr("INTERVAL 24 HOURS"))
          )
          .where((F.col("tpep_pickup_datetime") >= start_lit) &
                 (F.col("tpep_pickup_datetime") <= end_lit))
    )

    df = df.withColumn("bin_15m", floor_to_15min(F.col("tpep_pickup_datetime")))

    agg = (
        df.groupBy("bin_15m")
          .agg(
              F.count(F.lit(1)).alias("trips_15m"),
              F.countDistinct("PULocationID").alias("unique_PUs_15m")
          )
    )

    # fill missing 15-min bins within [start,end]
    cal = (spark.createDataFrame([(args.start, args.end)], ["start_ts", "end_ts"])
                 .select(F.explode(F.sequence(F.col("start_ts").cast("timestamp"),
                                             F.col("end_ts").cast("timestamp"),
                                             F.expr("interval 15 minutes"))).alias("bin_15m")))

    full = (
        cal.join(agg, "bin_15m", "left")
           .select(
               "bin_15m",
               F.coalesce("trips_15m", F.lit(0)).cast("long").alias("trips_15m"),
               F.coalesce("unique_PUs_15m", F.lit(0)).cast("long").alias("unique_PUs_15m"),
           )
           .orderBy("bin_15m")
    )

    full.write.mode("overwrite").parquet(OUT_TAXI_15M)
    print(f"[OK] Taxi 15-min → {OUT_TAXI_15M}")
    spark.stop()

if __name__ == "__main__":
    main()
