import argparse
import numpy as np
import pandas as pd
import torch
import os

from utils import load_checkpoint

def main(args):
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    load_scaler = meta["load_scaler"]
    price_scaler = meta["price_scaler"]

    # Load standardized forecasts saved by forecast.py
    data = np.load(args.std_npz, allow_pickle=True)
    ts_raw = data["timestamps"]

    # --- Robustly coerce numpy.str_ → Python str → pandas Timestamps
    # Works regardless of whether the array is dtype('<U...'), object, or bytes
    ts_str = np.asarray(ts_raw).astype(str)
    timestamps = pd.to_datetime(ts_str, errors="coerce")
    if timestamps.isna().any():
        bad = ts_str[pd.isna(timestamps)]
        raise ValueError(f"Unparseable timestamps found: {bad[:5]}")
    timestamps = timestamps.to_pydatetime().tolist()

    scenarios = data["scenarios"]  # [S, 2, T] standardized
    real = data["real"]            # [2, T] standardized
    load_col = str(data["load_col"])
    price_col = str(data["price_col"])

    S, _, T = scenarios.shape

    # ---- inverse transform back to real units ----
    # LOAD
    scen_load_real = load_scaler.inverse_transform(
        scenarios[:, 0, :].reshape(-1, 1)
    ).reshape(S, T)
    real_load_real = load_scaler.inverse_transform(
        real[0, :].reshape(-1, 1)
    ).reshape(T)

    # USEP (price)
    scen_price_real = price_scaler.inverse_transform(
        scenarios[:, 1, :].reshape(-1, 1)
    ).reshape(S, T)
    real_price_real = price_scaler.inverse_transform(
        real[1, :].reshape(-1, 1)
    ).reshape(T)

    # ---- tidy CSV with all scenarios + real ----
    rows = []
    for t_idx, ts in enumerate(timestamps):
        rows.append({"timestamp": ts, "series": "LOAD", "scenario": -1, "value": float(real_load_real[t_idx])})
        rows.append({"timestamp": ts, "series": "USEP", "scenario": -1, "value": float(real_price_real[t_idx])})
        for s in range(S):
            rows.append({"timestamp": ts, "series": "LOAD", "scenario": s, "value": float(scen_load_real[s, t_idx])})
            rows.append({"timestamp": ts, "series": "USEP", "scenario": s, "value": float(scen_price_real[s, t_idx])})

    df = pd.DataFrame(rows).sort_values(["timestamp", "series", "scenario"])
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, f"forecast_{args.tag}_real.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved inverse-transformed tidy CSV → {out_csv}")

    # ---- summary CSV (median, p10, p90, and real) ----
    med_load = np.median(scen_load_real, axis=0)
    p10_load = np.percentile(scen_load_real, 10, axis=0)
    p90_load = np.percentile(scen_load_real, 90, axis=0)

    med_price = np.median(scen_price_real, axis=0)
    p10_price = np.percentile(scen_price_real, 10, axis=0)
    p90_price = np.percentile(scen_price_real, 90, axis=0)

    df_sum = pd.DataFrame({
        "timestamp": timestamps,
        "med_LOAD": med_load,
        "p10_LOAD": p10_load,
        "p90_LOAD": p90_load,
        "med_USEP": med_price,
        "p10_USEP": p10_price,
        "p90_USEP": p90_price,
        "real_LOAD": real_load_real,
        "real_USEP": real_price_real,
    })
    sum_csv = os.path.join(args.outdir, f"forecast_{args.tag}_summary.csv")
    df_sum.to_csv(sum_csv, index=False)
    print(f"Saved summary CSV → {sum_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--std_npz", type=str, required=True)   # from forecast.py
    ap.add_argument("--outdir", type=str, default="outputs_forecast")
    ap.add_argument("--tag", type=str, default="day")
    args = ap.parse_args()
    main(args)
