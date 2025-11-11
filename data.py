

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

# ----- simple per-channel standard scaler (kept here to avoid circular imports) -----
class StandardScaler1D:
    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1) if x.ndim == 1 else np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1) if x.ndim == 1 else np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1) if x.ndim == 1 else np.asarray(x, dtype=np.float32)
        return x * self.std_ + self.mean_


# ----- dataset -----
class LoadPriceDataset(Dataset):
    """
    Returns tuples: (hist[T_h, 2], target[T_f, 2], cal_hist[T_h, F], cal_fut[T_f, F])
    where F is 4 if use_calendar, else 0. No None objects are returned.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        load_col: str,
        price_col: str,
        hist_len: int = 48,
        seq_len: int = 48,
        use_calendar: bool = False,
        load_scaler: Optional[StandardScaler1D] = None,
        price_scaler: Optional[StandardScaler1D] = None,
    ):
        self.df = df.sort_values(time_col).reset_index(drop=True)
        self.time_col = time_col
        self.load_col = load_col
        self.price_col = price_col
        self.hist_len = hist_len
        self.seq_len = seq_len
        self.use_calendar = use_calendar

        # Arrays
        self.load = self.df[load_col].values.astype(np.float32)
        self.price = self.df[price_col].values.astype(np.float32)

        # Scalers (shared between train/val)
        self.load_scaler = load_scaler or StandardScaler1D().fit(self.load)
        self.price_scaler = price_scaler or StandardScaler1D().fit(self.price)

        self.load_z = self.load_scaler.transform(self.load).astype(np.float32).squeeze()
        self.price_z = self.price_scaler.transform(self.price).astype(np.float32).squeeze()

        # Calendar features (sin/cos of hour-of-day and day-of-week)
        if self.use_calendar:
            t = pd.to_datetime(self.df[time_col])
            # infer samples per day from spacing (fallback: 48)
            if len(t) >= 2:
                delta_sec = max(1.0, (t.iloc[1] - t.iloc[0]).total_seconds())
                spd = int(round(86400.0 / delta_sec))
            else:
                spd = 48
            if spd < 4:
                spd = 24
            idx = np.arange(len(t)) % spd
            hod_sin = np.sin(2 * np.pi * idx / spd)
            hod_cos = np.cos(2 * np.pi * idx / spd)
            dow = t.dt.weekday.to_numpy()
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)
            self.cal_feats = np.stack([hod_sin, hod_cos, dow_sin, dow_cos], axis=1).astype(np.float32)
        else:
            self.cal_feats = None

        # number of windows
        self.length = max(0, len(self.df) - (self.hist_len + self.seq_len) + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        s = idx
        h0, h1 = s, s + self.hist_len
        t0, t1 = h1, h1 + self.seq_len

        # history (H, 2): [load, price]
        hist = np.stack([self.load_z[h0:h1], self.price_z[h0:h1]], axis=1).astype(np.float32)
        # target  (T, 2)
        target = np.stack([self.load_z[t0:t1], self.price_z[t0:t1]], axis=1).astype(np.float32)

        if self.use_calendar and self.cal_feats is not None:
            cal_hist = self.cal_feats[h0:h1]  # (H, 4)
            cal_fut = self.cal_feats[t0:t1]   # (T, 4)
        else:
            # Empty feature tensors with correct time lengths, zero feature dims.
            cal_hist = np.empty((self.hist_len, 0), dtype=np.float32)
            cal_fut  = np.empty((self.seq_len, 0), dtype=np.float32)

        return (
            torch.from_numpy(hist),            # (H, 2)
            torch.from_numpy(target),          # (T, 2)
            torch.from_numpy(cal_hist),        # (H, F)  F∈{0,4}
            torch.from_numpy(cal_fut),         # (T, F)
        )


# ----- helpers -----
def _maybe_build_timestamp(
    df: pd.DataFrame,
    time_col: str,
    date_col: Optional[str],
    period_col: Optional[str],
    periods_per_day: int,
):
    """
    If time_col exists -> return df, time_col.
    Else if date_col & period_col exist -> build a timestamp:
        timestamp = to_datetime(DATE) + (PERIOD-1) * (1440/periods_per_day) minutes
    Returns (df_with_timestamp, resolved_time_col).
    """
    if time_col in df.columns:
        return df, time_col

    if date_col and period_col and (date_col in df.columns) and (period_col in df.columns):
        base = pd.to_datetime(df[date_col])
        step_minutes = int(round(1440 / max(1, periods_per_day)))
        # PERIOD is 1-based in your file
        offset = pd.to_timedelta((df[period_col].astype(int) - 1) * step_minutes, unit="m")
        df = df.copy()
        df["__timestamp"] = base + offset
        return df, "__timestamp"

    raise KeyError(
        f"time_col '{time_col}' not found and could not derive from date_col/period_col."
    )


# ----- main loader factory -----
def make_loaders(
    csv_path: str,
    time_col: str,
    load_col: str,
    price_col: str,
    hist_len: int,
    seq_len: int,
    batch: int,
    val_ratio: float = 0.1,
    use_calendar: bool = False,
    num_workers: int = 0,
    date_col: Optional[str] = None,
    period_col: Optional[str] = None,
    periods_per_day: int = 48,
):
    df = pd.read_csv(csv_path)

    # Build a timestamp when needed
    df, resolved_time = _maybe_build_timestamp(df, time_col, date_col, period_col, periods_per_day)

    # Keep only needed cols and normalize the name 'timestamp'
    df = df[[resolved_time, load_col, price_col]].dropna().copy()
    df = df.rename(columns={resolved_time: "timestamp"})

    # time-based split
    n = len(df)
    n_val = int(n * val_ratio)
    df_tr = df.iloc[: n - n_val].copy()
    df_va = df.iloc[n - n_val :].copy()

    # shared scalers
    load_scaler = StandardScaler1D().fit(df_tr[load_col].values.astype(np.float32))
    price_scaler = StandardScaler1D().fit(df_tr[price_col].values.astype(np.float32))

    dtr = LoadPriceDataset(
        df_tr, "timestamp", load_col, price_col,
        hist_len, seq_len, use_calendar, load_scaler, price_scaler
    )
    dva = LoadPriceDataset(
        df_va, "timestamp", load_col, price_col,
        hist_len, seq_len, use_calendar, load_scaler, price_scaler
    )

    tr_loader = DataLoader(
        dtr, batch_size=batch, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=False  # False is safer on CPU/Colab
    )
    va_loader = DataLoader(
        dva, batch_size=batch, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=False
    )

    meta = {
        "load_scaler": load_scaler,
        "price_scaler": price_scaler,
        "n_train_rows": len(df_tr),
        "n_val_rows": len(df_va),
    }
    return tr_loader, va_loader, meta
