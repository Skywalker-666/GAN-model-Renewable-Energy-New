
import torch
from torch import autograd

# ==== ADD AT TOP OF train.py ====
import csv, time
from collections import defaultdict

class LossLogger:
    def __init__(self, outdir):
        self.step_rows = []
        self.epoch_rows = []
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.step_path = os.path.join(outdir, "loss_steps.csv")
        self.epoch_path = os.path.join(outdir, "loss_epochs.csv")
        # headers
        with open(self.step_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time","epoch","step","loss_G","loss_C","l1","adv","gp"])
        with open(self.epoch_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time","epoch","rmse_load","rmse_price","mae_load","mae_price","val_score","best_val"])

    def log_step(self, epoch, step, loss_G, loss_C=None, l1=None, adv=None, gp=None):
        with open(self.step_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()), epoch, step,
                        float(loss_G) if loss_G is not None else "",
                        float(loss_C) if loss_C is not None else "",
                        float(l1) if l1 is not None else "",
                        float(adv) if adv is not None else "",
                        float(gp) if gp is not None else ""])

    def log_epoch(self, epoch, rmse_load, rmse_price, mae_load, mae_price, val_score, best_val):
        with open(self.epoch_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()), epoch,
                        float(rmse_load), float(rmse_price),
                        float(mae_load), float(mae_price),
                        float(val_score), float(best_val)])



def gradient_penalty(critic, real, fake, h, ch, cf, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    inter = real * alpha + fake * (1 - alpha)
    inter.requires_grad_(True)
    d_inter = critic(inter, h, ch, cf)
    grad = autograd.grad(
        outputs=d_inter,
        inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad = grad.reshape(B, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp
