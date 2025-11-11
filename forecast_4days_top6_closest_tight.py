# =====================================
# forecast_4days_top6_closest_tight.py
# Ultra-tight selection: hug the real curve over 4 days
# =====================================
import os, sys, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ----------------- small utils -----------------
def _torch_load_any(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def _build_timestamp(df, time_col, date_col, period_col, ppd, LOAD_COL, PRICE_COL):
    if time_col and (time_col in df.columns):
        tcol = time_col
    elif (date_col in df.columns) and (period_col in df.columns):
        base = pd.to_datetime(df[date_col], errors="coerce")
        step = int(round(1440 / max(1, ppd)))
        off = pd.to_timedelta((df[period_col].astype(int) - 1) * step, unit="m")
        df = df.copy(); df["__timestamp"] = base + off; tcol = "__timestamp"
    else:
        raise KeyError("Provide time_col or (date_col & period_col).")
    df = df[[tcol, LOAD_COL, PRICE_COL]].dropna().rename(columns={tcol:"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def _pick_4day_window(df, start_date, H, T, days, ppd):
    ts = df["timestamp"]
    D0 = pd.Timestamp(pd.to_datetime(start_date).normalize())
    all_days = ts.dt.normalize().unique()
    if D0 not in all_days:
        bef = all_days[all_days < D0]; aft = all_days[all_days > D0]
        D0 = (bef.max() if len(bef) else aft.min())
        print(f"[info] target not present; using nearest → {D0.date()}")
    mask0 = (ts.dt.normalize() == D0)
    idx0 = np.where(mask0)[0]
    if len(idx0) == 0: raise ValueError("No rows for chosen start day")
    s = idx0[0]
    if s < H: raise ValueError("Not enough history before start day")

    end_day = D0 + pd.Timedelta(days=days-1)
    mask4 = (ts.dt.normalize() >= D0) & (ts.dt.normalize() <= end_day)
    idxs = np.where(mask4)[0]
    if len(idxs) < days*T: raise ValueError(f"Need full {days}×{T} rows after start day")

    hist = df.iloc[s-H:s].copy()
    fut = df.iloc[idxs[0]: idxs[0] + days*T].copy()
    xlabels = fut["timestamp"].dt.strftime("%d %b %H:%M").to_numpy()
    day_breaks = [i*T for i in range(1, days)]
    date_str = f"{str(D0.date())} to {str(end_day.date())}"
    return hist, fut, date_str, xlabels, day_breaks

def _inv_scale(arr, scaler):  # arr (S,T) or (T,)
    x = arr.reshape(-1, 1)
    return scaler.inverse_transform(x).reshape(arr.shape)

def _mae(a, b): return float(np.mean(np.abs(a - b)))

def _rbf_noise(T, length_scale=6.0, sigma=0.10, seed=None, device="cpu"):
    rng = np.random.default_rng(seed)
    x = np.arange(T)[:, None]
    d2 = (x - x.T)**2
    K = np.exp(-d2 / (2.0 * max(1e-6, length_scale)**2))
    K += 1e-6*np.eye(T)
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(T)
    y = L @ z
    y = y / (np.std(y) + 1e-8) * sigma
    return torch.from_numpy(y.astype(np.float32)).to(device)

def _time_warp(curve, strength=0.03):
    if strength <= 0: return curve
    T = curve.shape[0]
    t = np.linspace(0,1,T)
    freqs = np.array([1,2,3]); phases = np.random.rand(len(freqs))*2*np.pi
    warp = (sum(np.cos(2*np.pi*freqs[i]*t + phases[i]) for i in range(len(freqs))) / len(freqs))
    warp = warp / (np.max(np.abs(warp))+1e-8) * strength
    delta = np.exp(warp)
    s = np.cumsum(delta); s = (s - s.min())/(s.max()-s.min())
    x = np.linspace(0,1,T)
    return np.interp(x, s, curve)

def _plot_4days(xlabels, real, curves, title, ylabel, path, day_breaks):
    plt.figure(figsize=(14,5))
    for c in curves:
        plt.plot(c, linewidth=1.6, alpha=0.95)
    plt.plot(real, color="black", linewidth=3.0, label=f"Actual {ylabel}")
    step = max(1, len(xlabels)//16)
    plt.xticks(np.arange(0, len(xlabels), step), xlabels[::step], rotation=45, ha="right")
    for db in day_breaks:
        plt.axvline(db, linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title(title); plt.xlabel("Time"); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=170); plt.show()
    print(f"✅ Saved: {path}")

def _select_topk_closest(S, real, top_k=6):
    mae_all = np.mean(np.abs(S - real), axis=1)
    order = np.argsort(mae_all)
    return order[:min(top_k, len(order))], mae_all

def _stack_channels_to_real(S_std, load_s, price_s):
    S_load_real = _inv_scale(S_std[:,0,:], load_s)
    S_price_real = _inv_scale(S_std[:,1,:], price_s)
    return S_load_real, S_price_real

def _generate_candidates(G, ta, h, ch, cf, base, *,
                         n_scenarios=12000, z_std=0.15, residual_temp=0.85,
                         gp_len=6.0, gp_scale=0.10, dropout_p=0.03, device="cpu"):
    z_dim = ta.get("z_dim", 64)
    T = base.size(-1)
    cand = []
    z_bank = []
    with torch.no_grad():
        for _ in range(n_scenarios):
            z = torch.randn(1, z_dim, device=device) * z_std
            res = G(z, h, ch, cf) * residual_temp
            if dropout_p > 0:
                res = F.dropout(res, p=dropout_p, training=True)
            y_std = (base + res).squeeze(0)  # (2, T)
            for c in range(2):
                gp = _rbf_noise(T, length_scale=gp_len, sigma=gp_scale, device=device)
                y_std[c] = y_std[c] + gp
            cand.append(y_std.cpu().numpy())
            z_bank.append(z.squeeze(0).cpu().numpy())
    return np.stack(cand, axis=0), np.stack(z_bank, axis=0)

# ----------------- main (tight closeness) -----------------
def plot_topk_scenarios_4days_close(
    checkpoint_path,
    csv_path,
    start_date,                 # e.g., "2021-12-01"
    outdir="outputs_4days_close",
    load_col="LOAD",
    price_col="USEP",
    hist_len=48,
    seq_len=48,
    days=4,
    periods_per_day=48,
    # pool (tight)
    n_scenarios=12000,
    z_std=0.15,
    residual_temp=0.85,
    gp_len=6.0, gp_scale=0.10,
    warp_strength=0.03,
    dropout_p=0.03,
    # local refinement
    refine_topM=400,
    refine_tries=4,
    refine_z_std=0.05,
    # final selection
    top_k=6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = _torch_load_any(checkpoint_path, device)
    ta = ck["args"]

    from models import Generator
    G = Generator(z_dim=ta.get("z_dim", 64),
                  hist_len=hist_len, seq_len=seq_len,
                  use_calendar=ta.get("use_calendar", False)).to(device)
    G.load_state_dict(ck["G"]); G.eval()

    load_s, price_s = ck["meta"]["load_scaler"], ck["meta"]["price_scaler"]

    df = pd.read_csv(csv_path)
    df = _build_timestamp(df,
                          ta.get("time_col", None),
                          ta.get("date_col", "DATE"),
                          ta.get("period_col", "PERIOD"),
                          periods_per_day,
                          load_col, price_col)

    hist, fut4, date_span, xlabels, day_breaks = _pick_4day_window(
        df, start_date, hist_len, seq_len, days, periods_per_day
    )

    # Real standardized & real arrays across 4 days
    real_L_std = load_s.transform(fut4[load_col].values.astype(np.float32)).reshape(-1)
    real_P_std = price_s.transform(fut4[price_col].values.astype(np.float32)).reshape(-1)

    def _prep_history(history_df):
        hL = load_s.transform(history_df[load_col].values.astype(np.float32)).reshape(-1)
        hP = price_s.transform(history_df[price_col].values.astype(np.float32)).reshape(-1)
        h_np = np.stack([hL, hP], axis=0).astype(np.float32)
        h = torch.from_numpy(h_np[None, :, :]).to(device)  # (1,2,H)
        base = h[:, :, -seq_len:]
        if ta.get("use_calendar", False):
            ch = torch.zeros((1,4,hist_len), dtype=torch.float32, device=device)
            cf = torch.zeros((1,4,seq_len), dtype=torch.float32, device=device)
        else:
            ch = torch.empty((1,0,hist_len), dtype=torch.float32, device=device)
            cf = torch.empty((1,0,seq_len), dtype=torch.float32, device=device)
        return h, ch, cf, base

    # Generate per-day pools (std) and concatenate along time -> (S,2,4T)
    S_std = None
    z_bank_days = []
    for d in range(days):
        s = d * seq_len
        hist_df = pd.concat([hist, fut4.iloc[:s]]).iloc[-hist_len:]
        h, ch, cf, base = _prep_history(hist_df)
        Sd, z_bank = _generate_candidates(
            G, ta, h, ch, cf, base,
            n_scenarios=n_scenarios,
            z_std=z_std, residual_temp=residual_temp,
            gp_len=gp_len, gp_scale=gp_scale,
            dropout_p=dropout_p, device=device
        )
        S_std = Sd if S_std is None else np.concatenate([S_std, Sd], axis=2)
        z_bank_days.append(z_bank)

    # Convert to REAL for 4-day horizon scoring
    S_load, S_price = _stack_channels_to_real(S_std, load_s, price_s)
    real_L = _inv_scale(real_L_std, load_s)
    real_P = _inv_scale(real_P_std, price_s)

    # Tiny warp (optional)
    def _warp_batch(M, strength):
        if strength <= 0: return M
        out = np.empty_like(M)
        for i in range(M.shape[0]):
            out[i] = _time_warp(M[i], strength=strength)
        return out
    S_load = _warp_batch(S_load, warp_strength)
    S_price = _warp_batch(S_price, warp_strength)

    # Stage 1: best M by MAE
    idxL_M, maeL_all = _select_topk_closest(S_load, real_L, top_k=refine_topM)
    idxP_M, maeP_all = _select_topk_closest(S_price, real_P, top_k=refine_topM)

    # Stage 2: local refine around each seed (per day), choose best neighbor per day, re-concat -> pool
    def _refine_pool(idx_M, real_curve):
        refined_std = []
        for idx in idx_M:
            day_std_concat = None
            for d in range(days):
                z0 = z_bank_days[d][idx]  # (z_dim,)
                s = d * seq_len
                hist_df = pd.concat([hist, fut4.iloc[:s]]).iloc[-hist_len:]
                h, ch, cf, base = _prep_history(hist_df)

                local_cands = []
                with torch.no_grad():
                    for _ in range(refine_tries):
                        z = torch.from_numpy(
                            (z0 + np.random.normal(0, refine_z_std, size=z0.shape)).astype(np.float32)
                        )[None, :].to(device)
                        res = G(z, h, ch, cf) * residual_temp
                        if dropout_p > 0:
                            res = F.dropout(res, p=dropout_p, training=True)
                        y_std = (base + res).squeeze(0)  # (2,T)
                        for c in range(2):
                            gp = _rbf_noise(seq_len, length_scale=gp_len, sigma=gp_scale*0.5, device=device)
                            y_std[c] = y_std[c] + gp
                        local_cands.append(y_std.cpu().numpy())
                local_cands = np.stack(local_cands, axis=0)  # (R,2,T)
                L_real, P_real = _stack_channels_to_real(local_cands, load_s, price_s)
                seg_real_L = real_L[d*seq_len:(d+1)*seq_len]
                seg_real_P = real_P[d*seq_len:(d+1)*seq_len]
                mae_local = np.mean(np.abs(L_real - seg_real_L), axis=1) + \
                            np.mean(np.abs(P_real - seg_real_P), axis=1)
                best_local = int(np.argmin(mae_local))
                best_day_std = local_cands[best_local][None, ...]
                day_std_concat = best_day_std if day_std_concat is None else np.concatenate([day_std_concat, best_day_std], axis=2)
            refined_std.append(day_std_concat.squeeze(0))
        refined_std = np.stack(refined_std, axis=0)  # (M,2,4T)
        R_load, R_price = _stack_channels_to_real(refined_std, load_s, price_s)
        return refined_std, R_load, R_price

    refined_std_L, refined_load_L, refined_price_L = _refine_pool(idxL_M, real_L)
    refined_std_P, refined_load_P, refined_price_P = _refine_pool(idxP_M, real_P)

    # Merge original best-M and refined, pick top-k by MAE (per channel)
    pool_load = np.concatenate([S_load[idxL_M], refined_load_L], axis=0)
    pool_price = np.concatenate([S_price[idxP_M], refined_price_P], axis=0)

    idxL_final, _ = _select_topk_closest(pool_load, real_L, top_k=top_k)
    idxP_final, _ = _select_topk_closest(pool_price, real_P, top_k=top_k)

    # Plot & stats
    os.makedirs(outdir, exist_ok=True)
    p1 = os.path.join(outdir, f"{load_col}_top{top_k}_CLOSE_{date_span.replace(' ','_')}.png")
    p2 = os.path.join(outdir, f"{price_col}_top{top_k}_CLOSE_{date_span.replace(' ','_')}.png")

    _plot_4days(xlabels, real_L, [pool_load[i] for i in idxL_final],
                f"{load_col} — Real + {top_k} closest scenarios | {date_span}",
                load_col, p1, day_breaks)

    _plot_4days(xlabels, real_P, [pool_price[i] for i in idxP_final],
                f"{price_col} — Real + {top_k} closest scenarios | {date_span}",
                price_col, p2, day_breaks)

    print(f"[stats] final avg MAE — LOAD: {np.mean([_mae(pool_load[i], real_L) for i in idxL_final]):.4f}, "
          f"USEP: {np.mean([_mae(pool_price[i], real_P) for i in idxP_final]):.4f}")

    return {"span": date_span, "load_png": p1, "usep_png": p2, "outdir": outdir}

# ----------------- example call (edit paths/dates) -----------------
if __name__ == "__main__":
    result = plot_topk_scenarios_4days_close(
        checkpoint_path="checkpoints_residual/ckpt_best_calib.pt",
        csv_path="Real_usep_load_pv_rp.csv",
        start_date="2021-11-20",
        # pool tightness
        n_scenarios=12000,
        z_std=0.15,
        residual_temp=0.85,
        gp_len=6.0, gp_scale=0.10,
        warp_strength=0.03,
        dropout_p=0.03,
        # refinement
        refine_topM=400,
        refine_tries=4,
        refine_z_std=0.05,
        # final
        top_k=6
    )
    print(result)
