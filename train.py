import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from data import make_loaders
from models import Generator, Critic
from losses import gradient_penalty
from utils import seed_everything, save_checkpoint, rmse, mae


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tr_loader, va_loader, meta = make_loaders(
        csv_path=args.csv_path,
        time_col=args.time_col,
        load_col=args.load_col,
        price_col=args.price_col,
        hist_len=args.hist_len,
        seq_len=args.seq_len,
        batch=args.batch,
        val_ratio=args.val_ratio,
        use_calendar=args.use_calendar,
        num_workers=args.num_workers,
        date_col=args.date_col,
        period_col=args.period_col,
        periods_per_day=args.periods_per_day
    )

    G = Generator(z_dim=args.z_dim, hist_len=args.hist_len, seq_len=args.seq_len,
                  use_calendar=args.use_calendar).to(device)
    C = Critic(hist_len=args.hist_len, seq_len=args.seq_len,
               use_calendar=args.use_calendar).to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=args.lr_c, betas=(0.5, 0.9))

    best_val = float('inf')
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        C.train(); G.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, batch in enumerate(pbar):
            # batch: hist(B,H,2) target(B,T,2) cal_hist(B,H,F) cal_fut(B,T,F)
            h, y_real, ch, cf = batch
            h = h.permute(0, 2, 1).to(device)         # [B,2,H]
            y_real = y_real.permute(0, 2, 1).to(device)  # [B,2,T]
            ch = ch.permute(0, 2, 1).to(device) if ch is not None else None  # [B,F,H]
            cf = cf.permute(0, 2, 1).to(device) if cf is not None else None  # [B,F,T]

            # -------- baseline: yesterday's pattern (last H=48 from history) --------
            base = h[:, :, -args.seq_len:]            # [B,2,T] in standardized units

            # ===== Critic (skip during warmup) =====
            do_adv = epoch > args.warmup_l1
            if do_adv:
                for _ in range(args.c_iters):
                    z = torch.randn(h.size(0), args.z_dim, device=device)
                    with torch.no_grad():
                        res_det = G(z, h, ch, cf)     # Δy
                        y_fake_det = base + res_det   # final sample
                    C_real = C(y_real, h, ch, cf).mean()
                    C_fake = C(y_fake_det, h, ch, cf).mean()
                    gp = gradient_penalty(C, y_real, y_fake_det, h, ch, cf, device)
                    loss_C = -(C_real - C_fake) + args.gp_lambda * gp
                    opt_C.zero_grad(set_to_none=True)
                    loss_C.backward()
                    opt_C.step()

            # ===== Generator =====
            z = torch.randn(h.size(0), args.z_dim, device=device)
            res = G(z, h, ch, cf)                     # Δy
            y_fake = base + res                       # final

            # Strong supervised term (pulls to real)
            l1 = (y_fake - y_real).abs().mean()

            if do_adv:
                adv_loss = -C(y_fake, h, ch, cf).mean()
                G_loss = args.adv_weight * adv_loss + args.lambda_l1 * l1
            else:
                # warmup: pure L1 for stability/proximity
                G_loss = args.lambda_l1 * l1

            opt_G.zero_grad(set_to_none=True)
            G_loss.backward()
            opt_G.step()

        # ===== Validation =====
        G.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for batch in va_loader:
                h, y_real, ch, cf = batch
                h = h.permute(0, 2, 1).to(device)
                y_real = y_real.permute(0, 2, 1).to(device)
                ch = ch.permute(0, 2, 1).to(device) if ch is not None else None
                cf = cf.permute(0, 2, 1).to(device) if cf is not None else None

                base = h[:, :, -args.seq_len:]        # [B,2,T]
                B = h.size(0)
                samples = []
                for _ in range(args.val_samples):
                    z = torch.randn(B, args.z_dim, device=device)
                    res = G(z, h, ch, cf)
                    samples.append((base + res).unsqueeze(0))
                S = torch.cat(samples, dim=0)         # [S,B,2,T]
                y_med = S.median(dim=0).values        # [B,2,T]

                y_true_all.append(y_real.cpu().numpy())
                y_pred_all.append(y_med.cpu().numpy())

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)

        rmse_load = rmse(y_true[:, 0, :], y_pred[:, 0, :])
        rmse_price = rmse(y_true[:, 1, :], y_pred[:, 1, :])
        mae_load = mae(y_true[:, 0, :], y_pred[:, 0, :])
        mae_price = mae(y_true[:, 1, :], y_pred[:, 1, :])
        val_score = (rmse_load + rmse_price) / 2.0

        print(f"Val — RMSE load: {rmse_load:.4f}, RMSE price: {rmse_price:.4f} | "
              f"MAE load: {mae_load:.4f}, MAE price: {mae_price:.4f}")

        # Save last and best
        save_checkpoint({
            "G": G.state_dict(),
            "C": C.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_C": opt_C.state_dict(),
            "args": vars(args),
            "meta": meta
        }, args.outdir, "ckpt_last.pt")

        if val_score < best_val:
            best_val = val_score
            save_checkpoint({
                "G": G.state_dict(),
                "C": C.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_C": opt_C.state_dict(),
                "args": vars(args),
                "meta": meta,
                "best_val": best_val
            }, args.outdir, "ckpt_best.pt")
            print(f"Saved new best with score {best_val:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_path', type=str, required=True)
    # Either provide time_col, or provide (date_col, period_col, periods_per_day)
    ap.add_argument('--time_col', type=str, default='timestamp')
    ap.add_argument('--date_col', type=str, default=None)
    ap.add_argument('--period_col', type=str, default=None)
    ap.add_argument('--periods_per_day', type=int, default=48)

    ap.add_argument('--load_col', type=str, default='LOAD')
    ap.add_argument('--price_col', type=str, default='USEP')  # your CSV
    ap.add_argument('--hist_len', type=int, default=48)
    ap.add_argument('--seq_len', type=int, default=48)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--z_dim', type=int, default=64)
    ap.add_argument('--lr_g', type=float, default=2e-4)
    ap.add_argument('--lr_c', type=float, default=2e-4)
    ap.add_argument('--c_iters', type=int, default=5)
    ap.add_argument('--gp_lambda', type=float, default=10.0)
    ap.add_argument('--val_samples', type=int, default=30)
    ap.add_argument('--lambda_l1', type=float, default=20.0)   # ↑ strong proximity
    ap.add_argument('--adv_weight', type=float, default=1.0)
    ap.add_argument('--warmup_l1', type=int, default=3)        # epochs of pure L1
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use_calendar', action='store_true')
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--outdir', type=str, default='checkpoints')
    args = ap.parse_args()
    train(args)
