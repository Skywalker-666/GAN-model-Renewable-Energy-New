import argparse
import numpy as np
import torch
import pandas as pd

from data import make_loaders
from models import Generator            # <- fix: models, not model
from utils import load_checkpoint, rmse, mae


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # ---- Load checkpoint and pull training args to avoid shape mismatches
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    targs = ckpt["args"]
    z_dim       = targs.get("z_dim", args.z_dim)
    hist_len    = targs.get("hist_len", args.hist_len)
    seq_len     = targs.get("seq_len", args.seq_len)
    use_calendar= targs.get("use_calendar", args.use_calendar)
    # column names
    load_col    = targs.get("load_col", args.load_col)
    price_col   = targs.get("price_col", args.price_col)
    # time resolution
    time_col    = targs.get("time_col", args.time_col)
    date_col    = targs.get("date_col", args.date_col)
    period_col  = targs.get("period_col", args.period_col)
    ppd         = targs.get("periods_per_day", args.periods_per_day)
    # loader hyperparams
    batch       = targs.get("batch", args.batch)
    val_ratio   = targs.get("val_ratio", args.val_ratio)
    num_workers = targs.get("num_workers", args.num_workers)

    # ---- Build ONLY the validation loader (same split logic as training)
    _, va_loader, meta = make_loaders(
        csv_path=args.csv_path,
        time_col=time_col,
        load_col=load_col,
        price_col=price_col,
        hist_len=hist_len,
        seq_len=seq_len,
        batch=batch,
        val_ratio=val_ratio,
        use_calendar=use_calendar,
        num_workers=num_workers,
        date_col=date_col,
        period_col=period_col,
        periods_per_day=ppd
    )

    # ---- Model
    G = Generator(z_dim=z_dim, hist_len=hist_len, seq_len=seq_len, use_calendar=use_calendar).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # Optional affine calibration (standardized space) if present
    calib = ckpt.get("meta", {}).get("affine_calib", None)
    def apply_calib(y):  # y: [B,2,T] standardized
        if calib is None:
            return y
        aL, bL = calib["load"]["a"],  calib["load"]["b"]
        aP, bP = calib["price"]["a"], calib["price"]["b"]
        y2 = y.clone()
        y2[:,0,:] = y2[:,0,:]*aL + bL
        y2[:,1,:] = y2[:,1,:]*aP + bP
        return y2

    y_true_all, y_point_all = [], []

    # ---- Evaluate (median of samples) using residual + baseline
    with torch.no_grad():
        for batch in va_loader:
            h, y_real, ch, cf = batch                     # h:[B,H,2], y_real:[B,T,2]
            h  = h.permute(0, 2, 1).to(device)            # [B,2,H]
            y_real = y_real.permute(0, 2, 1).to(device)   # [B,2,T]
            ch = ch.permute(0, 2, 1).to(device) if (use_calendar and ch is not None) else None  # [B,4,H]
            cf = cf.permute(0, 2, 1).to(device) if (use_calendar and cf is not None) else None  # [B,4,T]

            base = h[:, :, -seq_len:]                     # previous 48, standardized

            B = h.size(0)
            samples = []
            for _ in range(args.n_samples):
                z = torch.randn(B, z_dim, device=device)
                res = G(z, h, ch, cf)                     # Δy
                y_fake = base + res                       # final standardized
                y_fake = apply_calib(y_fake)              # optional affine fix
                samples.append(y_fake.unsqueeze(0))
            S = torch.cat(samples, dim=0)                 # [S,B,2,T]
            y_med = S.median(dim=0).values                # [B,2,T]

            y_true_all.append(y_real.cpu().numpy())
            y_point_all.append(y_med.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)   # standardized
    y_pred = np.concatenate(y_point_all, axis=0)

    print("Validation metrics (standardized space):")
    print(f"RMSE LOAD : {rmse(y_true[:,0,:], y_pred[:,0,:]):.4f}")
    print(f"RMSE USEP : {rmse(y_true[:,1,:], y_pred[:,1,:]):.4f}")
    print(f"MAE  LOAD : {mae(y_true[:,0,:], y_pred[:,0,:]):.4f}")
    print(f"MAE  USEP : {mae(y_true[:,1,:], y_pred[:,1,:]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--csv_path', type=str, required=True)

    # Either provide time_col OR (date_col, period_col, periods_per_day)
    ap.add_argument('--time_col', type=str, default='timestamp')
    ap.add_argument('--date_col', type=str, default=None)
    ap.add_argument('--period_col', type=str, default=None)
    ap.add_argument('--periods_per_day', type=int, default=48)

    ap.add_argument('--load_col', type=str, default='LOAD')
    ap.add_argument('--price_col', type=str, default='USEP')   # <- your CSV uses USEP
    ap.add_argument('--hist_len', type=int, default=48)
    ap.add_argument('--seq_len', type=int, default=48)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--z_dim', type=int, default=64)
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--use_calendar', action='store_true')
    ap.add_argument('--num_workers', type=int, default=0)

    args = ap.parse_args()
    evaluate(args)
