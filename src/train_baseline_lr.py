# src/train_baseline_lr.py
import os, argparse, json
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

NYC_TZ = "America/New_York"

def _unique_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_horizons(hstr):
    hstr = str(hstr).strip().lower()
    if hstr == "short":
        return [15, 30, 60], True
    if hstr == "auto":
        return [], False
    raw = [x.strip() for x in hstr.split(",") if x.strip()]
    mins = []
    for x in raw:
        n = int(x)
        mins.append(n if (n >= 15 and n % 15 == 0) else n * 15)
    return sorted(set(mins)), True

def discover_label_minutes(df_cols):
    mins = set()
    for c in df_cols:
        cl = c.lower()
        if cl.startswith("y_t_plus_") and cl.endswith("m"):
            try:
                n = int(cl[len("y_t_plus_"):-1])
                if n > 0 and n % 15 == 0:
                    mins.add(n)
            except ValueError:
                pass
        if cl.startswith("y_tplus_"):
            try:
                steps = int(cl[len("y_tplus_"):])
                if steps > 0:
                    mins.add(steps * 15)
            except ValueError:
                pass
        if cl.startswith("y_tplus"):
            tail = cl[len("y_tplus"):]
            if tail.isdigit():
                mins.add(int(tail) * 15)
    return sorted(mins)

def find_label_col(df_cols, minutes):
    steps = minutes // 15
    candidates = [f"y_t_plus_{minutes}m", f"y_tplus_{steps}", f"y_tplus{steps}"]
    for c in candidates:
        for dc in df_cols:
            if dc.lower() == c.lower():
                return dc
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/nyc-traffic-forecast/artifacts/features/supervised_15m"))
    ap.add_argument("--horizons", default="short")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--outdir", default=os.path.expanduser("~/nyc-traffic-forecast/artifacts/results"))
    ap.add_argument("--save_models", action="store_true", help="Save trained models to subfolder 'models'")
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("Baseline_LR_TaxiDemand")
             .config("spark.sql.session.timeZone", NYC_TZ)
             .getOrCreate())

    df = spark.read.parquet(args.data)

    if "bin_15m" not in df.columns and "ts" in df.columns:
        df = df.withColumn("bin_15m", F.col("ts"))
    if "bin_15m" not in df.columns:
        raise SystemExit("[ERROR] Missing time column")

    feature_cols = []
    feature_cols += [c for c in df.columns if c.startswith("lag_")]
    feature_cols += [c for c in df.columns if c.startswith("ma_")]
    for p in ("temp_C", "dew_C", "wind_mps", "precip_mm"):
        feature_cols += [c for c in df.columns if c.startswith(p)]
    for c in ["hour", "dow", "is_weekend", "is_rush_am", "is_rush_pm", "month"]:
        if c in df.columns:
            feature_cols.append(c)

    candidate_features = _unique_keep_order(
        [c for c in feature_cols if c not in ("trips_15m", "bin_15m", "ts")]
    )
    if not candidate_features:
        raise SystemExit("[ERROR] No features found")

    df = df.withColumn("ts_idx", F.col("bin_15m").cast("timestamp").cast("long"))
    stats = df.select(F.min("ts_idx").alias("min_ts"), F.max("ts_idx").alias("max_ts")).first()
    cutoff = stats.min_ts + (stats.max_ts - stats.min_ts) * (1.0 - args.test_frac)
    train = df.where(F.col("ts_idx") <= cutoff).cache()
    test  = df.where(F.col("ts_idx") >  cutoff).cache()

    print(f"[INFO] rows: total={df.count()} train={train.count()} test={test.count()}")
    print(f"[INFO] using {len(candidate_features)} features")

    mins_list, had_explicit = parse_horizons(args.horizons)
    if not had_explicit:
        mins_list = discover_label_minutes(df.columns)
        print(f"[INFO] auto-discovered horizons: {mins_list}")

    os.makedirs(args.outdir, exist_ok=True)
    pred_dir = os.path.join(args.outdir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_models:
        model_dir = os.path.join(args.outdir, "models")
        os.makedirs(model_dir, exist_ok=True)

    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    mae_eval = RegressionEvaluator(metricName="mae", predictionCol="prediction")
    rmse_eval = RegressionEvaluator(metricName="rmse", predictionCol="prediction")
    r2_eval   = RegressionEvaluator(metricName="r2",  predictionCol="prediction")

    results = []
    for m in mins_list:
        label_col = find_label_col(df.columns, minutes=m)
        if not label_col:
            print(f"[WARN] No label found for {m}m; skipping.")
            continue

        cols_needed = [label_col] + candidate_features + ["bin_15m"]
        tr = train.select(*cols_needed).na.drop()
        te = test.select(*cols_needed).na.drop()

        assembler = VectorAssembler(inputCols=candidate_features, outputCol="features_raw")
        tr = assembler.transform(tr)
        te = assembler.transform(te)

        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
        scaler_model = scaler.fit(tr)
        tr = scaler_model.transform(tr)
        te = scaler_model.transform(te)

        lr = LinearRegression(
            featuresCol="features",
            labelCol=label_col,
            predictionCol="prediction",
            elasticNetParam=0.5,
            regParam=0.05,
            maxIter=100
        )
        model = lr.fit(tr)
        if args.save_models:
            save_path = os.path.join(model_dir, f"lr_horizon_{m}m.model")
            model.write().overwrite().save(save_path)
            print(f"[INFO] Saved model: {save_path}")

        pred = (te.withColumnRenamed(label_col, "label")
                    .select("bin_15m", "label", "features"))
        pred = model.transform(pred).select("bin_15m", "label", "prediction")

        mae_v  = mae_eval.evaluate(pred)
        rmse_v = rmse_eval.evaluate(pred)
        r2_v   = r2_eval.evaluate(pred)

        print(f"[RESULT] horizon={m}m  MAE={mae_v:.3f}  RMSE={rmse_v:.3f}  R2={r2_v:.3f}")
        pred.write.mode("overwrite").parquet(os.path.join(pred_dir, f"horizon={m}m.parquet"))

        rec = {"horizon_min": m, "mae": float(mae_v), "rmse": float(rmse_v), "r2": float(r2_v)}
        results.append(rec)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    if results:
        import csv
        csv_path = os.path.join(args.outdir, "metrics_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["horizon_min", "mae", "rmse", "r2"])
            w.writeheader()
            for r in sorted(results, key=lambda x: x["horizon_min"]):
                w.writerow(r)
        print(f"[INFO] Wrote metrics to {metrics_path} and {csv_path}")
        print(f"[INFO] Predictions written under {pred_dir}")
    else:
        print("[ERROR] No models trained; check label columns.")

    spark.stop()

if __name__ == "__main__":
    main()

