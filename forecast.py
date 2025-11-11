import argparse, numpy as np, pandas as pd, torch, os
from utils import load_checkpoint
from models import Generator

def _resolve_timestamp(df, time_col, date_col, period_col, periods_per_day):
    if time_col and (time_col in df.columns): return df, time_col
    if (date_col in df.columns) and (period_col in df.columns):
        base = pd.to_datetime(df[date_col], errors="coerce")
        step = int(round(1440 / max(1, periods_per_day)))
        off  = pd.to_timedelta((df[period_col].astype(int) - 1) * step, unit="m")
        df = df.copy(); df["__timestamp"] = base + off
        return df, "__timestamp"
    raise KeyError("Provide --time_col OR (--date_col and --period_col).")

def _parse_date(s):
    ts = pd.to_datetime(s, errors="coerce");
    if pd.isna(ts): ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(ts): raise ValueError(f"Bad target_date {s}")
    return pd.Timestamp(ts.normalize())

def _hist_and_day(df, tcol, d, H, T, ppd):
    df = df.sort_values(tcol).reset_index(drop=True)
    ts = pd.to_datetime(df[tcol]); D = pd.Timestamp(d.normalize())
    avail = pd.Index(ts.dt.normalize().unique())
    if D not in avail:
        bef = avail[avail < D]; aft = avail[avail > D]
        D = (bef.max() if len(bef) else aft.min())
        print(f"[forecast] fallback to {D.date()}")
    m = ts.dt.normalize() == D; idx = np.where(m)[0]
    if len(idx)==0: raise ValueError("no rows for chosen day")
    s = idx[0];
    if s < H:   raise ValueError("not enough history")
    hist = df.iloc[s-H:s].copy(); day = df.iloc[s:s+T].copy()
    if len(day) < T: raise ValueError("need full 48")
    return hist, day, D.date()

def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ck  = load_checkpoint(a.checkpoint, map_location=dev)
    ta  = ck["args"]; H=ta["hist_len"]; T=ta["seq_len"]; use_cal=ta.get("use_calendar", False)
    G   = Generator(z_dim=ta["z_dim"], hist_len=H, seq_len=T, use_calendar=use_cal).to(dev)
    G.load_state_dict(ck["G"]); G.eval()

    df = pd.read_csv(a.csv_path)
    df, tcol = _resolve_timestamp(df, a.time_col, a.date_col, a.period_col, a.periods_per_day)
    df = df[[tcol, a.load_col, a.price_col]].dropna().copy()
    df = df.rename(columns={tcol:"timestamp"}); df["timestamp"]=pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    d = _parse_date(a.target_date)
    hist, fut, used = _hist_and_day(df, "timestamp", d, H, T, a.periods_per_day)

    meta = ck["meta"]; load_s = meta["load_scaler"]; price_s = meta["price_scaler"]
    # std history
    h_load  = load_s.transform(hist[a.load_col].values.astype(np.float32)).reshape(-1)
    h_price = price_s.transform(hist[a.price_col].values.astype(np.float32)).reshape(-1)
    h_np = np.stack([h_load, h_price], axis=0).astype(np.float32)  # (2,H)
    h = torch.from_numpy(h_np[None,:,:]).to(dev)                   # (1,2,H)

    # calendar channels (zeros to match training)
    if use_cal:
        ch = torch.zeros((1,4,H), dtype=torch.float32, device=dev)
        cf = torch.zeros((1,4,T), dtype=torch.float32, device=dev)
    else:
        ch = torch.empty((1,0,H), dtype=torch.float32, device=dev)
        cf = torch.empty((1,0,T), dtype=torch.float32, device=dev)

    base = h[:, :, -T:]  # (1,2,T) standardized

    # optional affine calibration in standardized space
    calib = meta.get("affine_calib", None)
    def apply_calib(y):
        if calib is None: return y
        aL, bL = calib["load"]["a"],  calib["load"]["b"]
        aP, bP = calib["price"]["a"], calib["price"]["b"]
        y2 = y.clone()
        y2[:,0,:] = y2[:,0,:]*aL + bL
        y2[:,1,:] = y2[:,1,:]*aP + bP
        return y2

    S = []
    with torch.no_grad():
        for _ in range(a.n_scenarios):
            z = torch.zeros(1, ta["z_dim"], device=dev) if a.z_std==0 else \
                torch.randn(1, ta["z_dim"], device=dev)*a.z_std
            res = G(z, h, ch, cf)        # Δy
            y   = base + res             # final std
            y   = apply_calib(y)
            S.append(y.squeeze(0).cpu().numpy())
    scenarios = np.stack(S, axis=0)  # [S,2,T]

    # real std for that day
    rL = load_s.transform(fut[a.load_col].values.astype(np.float32)).reshape(-1)
    rP = price_s.transform(fut[a.price_col].values.astype(np.float32)).reshape(-1)
    real = np.stack([rL, rP], axis=0).astype(np.float32)

    times = list(np.array(fut["timestamp"].dt.to_pydatetime()))
    os.makedirs(a.outdir, exist_ok=True)
    np.savez_compressed(
        f"{a.outdir}/forecast_{used}_std.npz",
        timestamps=np.array([t.isoformat() for t in times]),
        scenarios=scenarios,
        real=real,
        load_col=a.load_col, price_col=a.price_col
    )
    print(f"Saved standardized forecast for {used}.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--date_col", type=str, default="DATE")
    ap.add_argument("--period_col", type=str, default="PERIOD")
    ap.add_argument("--periods_per_day", type=int, default=48)
    ap.add_argument("--load_col", type=str, default="LOAD")
    ap.add_argument("--price_col", type=str, default="USEP")
    ap.add_argument("--target_date", type=str, required=True)
    ap.add_argument("--n_scenarios", type=int, default=20)
    ap.add_argument("--z_std", type=float, default=0.5, help="0 for deterministic mean")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--outdir", type=str, default="outputs_forecast")
    args = ap.parse_args(); main(args)
