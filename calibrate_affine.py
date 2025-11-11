import argparse, os, torch, numpy as np
from utils import load_checkpoint, save_checkpoint, seed_everything
from data import make_loaders
from models import Generator

def fit_affine(y_pred, y_true):
    # Solve y_true ≈ a * y_pred + b (least squares), vectorized over all horizons
    x = y_pred.reshape(-1, 1)             # (N,1)
    y = y_true.reshape(-1, 1)             # (N,1)
    X = np.concatenate([x, np.ones_like(x)], axis=1)  # [x, 1]
    # a,b = (X^T X)^-1 X^T y
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(beta[0,0]); b = float(beta[1,0])
    return a, b

def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    targs = ckpt["args"]
    hist_len = targs["hist_len"]; seq_len = targs["seq_len"]; use_calendar = targs.get("use_calendar", False)

    # data loaders with same args used in training
    _, va_loader, meta = make_loaders(
        csv_path=args.csv_path,
        time_col=targs.get("time_col"),
        load_col=targs["load_col"],
        price_col=targs["price_col"],
        hist_len=hist_len,
        seq_len=seq_len,
        batch=targs.get("batch", 128),
        val_ratio=targs.get("val_ratio", 0.1),
        use_calendar=use_calendar,
        num_workers=targs.get("num_workers", 0),
        date_col=targs.get("date_col"),
        period_col=targs.get("period_col"),
        periods_per_day=targs.get("periods_per_day", 48),
    )

    G = Generator(z_dim=targs["z_dim"], hist_len=hist_len, seq_len=seq_len,
                  use_calendar=use_calendar).to(device)
    G.load_state_dict(ckpt["G"]); G.eval()

    preds = []; trues = []
    with torch.no_grad():
        for h, y_real, ch, cf in va_loader:
            h  = h.permute(0,2,1).to(device)         # [B,2,H]
            y  = y_real.permute(0,2,1).to(device)    # [B,2,T]
            ch = ch.permute(0,2,1).to(device) if ch is not None else None
            cf = cf.permute(0,2,1).to(device) if cf is not None else None

            base = h[:, :, -seq_len:]
            z = torch.zeros(h.size(0), targs["z_dim"], device=device)  # deterministic mean-ish
            res = G(z, h, ch, cf)
            y_hat = base + res  # [B,2,T]
            preds.append(y_hat.cpu().numpy()); trues.append(y.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)   # [N,2,T] standardized
    y_true = np.concatenate(trues, axis=0)   # [N,2,T]

    # fit per-channel affine
    a_load, b_load   = fit_affine(y_pred[:,0,:], y_true[:,0,:])
    a_price, b_price = fit_affine(y_pred[:,1,:], y_true[:,1,:])

    print(f"[calibrate] LOAD:  a={a_load:.4f}, b={b_load:.4f}")
    print(f"[calibrate] USEP:  a={a_price:.4f}, b={b_price:.4f}")

    # stash into checkpoint meta
    meta = ckpt.get("meta", {})
    meta["affine_calib"] = {
        "load":  {"a": a_load,  "b": b_load},
        "price": {"a": a_price, "b": b_price}
    }
    ckpt["meta"] = meta

    outdir = os.path.dirname(args.checkpoint)
    save_checkpoint(ckpt, outdir, "ckpt_best_calib.pt")
    print(f"Saved calibrated checkpoint to {os.path.join(outdir,'ckpt_best_calib.pt')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
