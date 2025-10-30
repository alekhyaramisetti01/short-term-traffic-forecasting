# src/build_supervised.py
import os, argparse
from pyspark.sql import SparkSession, functions as F, Window

NYC_TZ = "America/New_York"

def _mins_list(spec: str):
    """
    Accepts:
      - "short" -> [15,30,60]
      - comma list of minutes (15,30,60) or steps (1,2,4). Mixed allowed.
    Returns a sorted unique list of minutes (multiples of 15).
    """
    spec = str(spec).strip().lower()
    if spec == "short":
        return [15, 30, 60]
    mins = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        n = int(tok)
        mins.append(n if (n >= 15 and n % 15 == 0) else n * 15)  # steps→minutes
    mins = sorted({m for m in mins if m > 0 and m % 15 == 0})
    return mins

def _ensure_bin_15m(df):
    """Make sure we have a timestamp column named bin_15m (aligned to 15-minute bins)."""
    if "bin_15m" in df.columns:
        df = df.withColumn("bin_15m", F.col("bin_15m").cast("timestamp"))
    elif "ts" in df.columns:
        df = df.withColumn("bin_15m", F.col("ts").cast("timestamp"))
    else:
        raise SystemExit("[ERROR] Need a time column 'bin_15m' or 'ts'.")
    # Snap to exact 15-min boundaries
    df = df.withColumn(
        "bin_15m",
        F.from_unixtime((F.unix_timestamp("bin_15m") / 900).cast("long") * 900).cast("timestamp")
    )
    return df

def _add_calendar(df):
    """Calendar/time-based features."""
    df = df.withColumn("bin_15m_local", F.from_utc_timestamp("bin_15m", NYC_TZ))
    df = df.withColumn("hour", F.hour("bin_15m_local"))
    df = df.withColumn("dow", F.dayofweek("bin_15m_local") - 1)  # 0=Sun → 0=Mon optional
    df = df.withColumn("month", F.month("bin_15m_local"))
    df = df.withColumn("is_rush_am", ((F.col("hour") >= 7) & (F.col("hour") <= 9)).cast("int"))
    df = df.withColumn("is_rush_pm", ((F.col("hour") >= 16) & (F.col("hour") <= 18)).cast("int"))
    df = df.withColumn("is_weekend", (F.col("dow").isin(5,6)).cast("int"))
    return df.drop("bin_15m_local")

def _add_lags_and_ma(df, target="trips_15m"):
    """Add lag and moving-average features for target variable."""
    w = Window.orderBy(F.col("bin_15m").cast("long"))
    for k in range(1, 97):  # 1 day = 96 steps (15 min each)
        df = df.withColumn(f"lag_{k}", F.lag(F.col(target), k).over(w))
    for wlen in [4, 8, 16, 32, 96]:  # 1h, 2h, 4h, 8h, 24h
        df = df.withColumn(f"ma_{wlen}", F.avg(F.col(target)).over(w.rowsBetween(-wlen + 1, 0)))
    return df

def _add_labels(df, target="trips_15m", mins_list=(15,30,60), emit_step_alias=True):
    """Create future label columns for given horizons."""
    w = Window.orderBy(F.col("bin_15m").cast("long"))
    for m in mins_list:
        steps = m // 15
        label_m = f"y_t_plus_{m}m"
        df = df.withColumn(label_m, F.lead(F.col(target), steps).over(w))
        if emit_step_alias:
            label_s = f"y_tplus_{steps}"
            df = df.withColumn(label_s, F.col(label_m))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
        default=os.path.expanduser("~/nyc-traffic-forecast/data/processed/merged/taxi_weather_15m.parquet"),
        help="Input merged parquet (must include trips_15m + weather + bin_15m)")
    ap.add_argument("--out",
        default=os.path.expanduser("~/nyc-traffic-forecast/artifacts/features/supervised_15m"),
        help="Output folder for supervised parquet data")
    ap.add_argument("--horizons", default="short",
        help="Label horizons: 'short' or comma list like '15,30,60' or steps '1,2,4'")
    ap.add_argument("--emit_step_alias", action="store_true",
        help="Also emit y_tplus_K aliases alongside y_t_plus_{K*15}m")
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("Build_Supervised_15m")
             .config("spark.sql.session.timeZone", NYC_TZ)
             .getOrCreate())

    print(f"[INFO] Reading merged data from: {args.src}")
    df = spark.read.parquet(args.src)
    df = _ensure_bin_15m(df)

    if "trips_15m" not in df.columns:
        raise SystemExit("[ERROR] Input must include 'trips_15m' column (taxi trip count).")

    df = (df.dropDuplicates(["bin_15m"])
            .orderBy(F.col("bin_15m").cast("long")))

    df = _add_calendar(df)
    df = _add_lags_and_ma(df, target="trips_15m")

    mins_list = _mins_list(args.horizons)
    df = _add_labels(df, target="trips_15m", mins_list=mins_list, emit_step_alias=args.emit_step_alias)

    label_cols = [f"y_t_plus_{m}m" for m in mins_list]
    if args.emit_step_alias:
        label_cols += [f"y_tplus_{m//15}" for m in mins_list]

    df_out = df.na.drop(subset=["trips_15m"] + label_cols)

    out_dir = args.out
    if os.path.exists(out_dir):
        print(f"[INFO] Clearing old output at: {out_dir}")
        import shutil
        shutil.rmtree(out_dir)

    df_out.write.mode("overwrite").parquet(out_dir)

    print(f"[OK] Supervised data written to: {out_dir}")
    print(f"[INFO] Horizons (minutes): {mins_list}")
    if args.emit_step_alias:
        print("[INFO] Step aliases y_tplus_K also written.")
    spark.stop()

if __name__ == "__main__":
    main()

