import torch
import torch.nn as nn
import torch.nn.functional as F

class TConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, p=None):
        super().__init__()
        pad = p if p is not None else (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResTCN(nn.Module):
    def __init__(self, c_in, channels=(64, 64, 64), k=3):
        super().__init__()
        layers, last = [], c_in
        for c in channels:
            layers += [TConvBlock(last, c, k=k)]
            last = c
        self.net = nn.Sequential(*layers)
        self.out_channels = last
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    """
    Predicts Δy (residual) given history & noise.
      z : [B, z_dim]
      h : [B, 2, H]      (standardized)
      ch: [B, 4, H] or empty
      cf: [B, 4, T] or empty (unused here, kept for API)
    Returns residual Δy: [B, 2, T] (standardized units)
    """
    def __init__(self, z_dim=64, hist_len=48, seq_len=48, use_calendar=False):
        super().__init__()
        self.z_dim, self.hist_len, self.seq_len = z_dim, hist_len, seq_len
        self.use_calendar = use_calendar

        cond_ch = 2 + (4 if use_calendar else 0)
        self.enc = ResTCN(c_in=cond_ch, channels=(64, 128, 128))
        self.fc_z = nn.Linear(z_dim, 128)
        self.fc_hist = nn.Linear(self.enc.out_channels, 128)
        self.fc = nn.Linear(256, 128)

        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 2, kernel_size=1)  # 2 channels residual
        )
        target_len = 2 * self.hist_len
        self.proj_len = nn.Linear(target_len, self.seq_len)

        # Scale residuals conservatively; learnable for flexibility
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, h, ch=None, cf=None):
        if self.use_calendar and ch is not None and ch.size(1) > 0:
            x = torch.cat([h, ch], dim=1)     # [B, 2(+4), H]
        else:
            x = h

        feat = self.enc(x).mean(dim=-1)       # [B, C]
        hz = torch.relu(self.fc_hist(feat))   # [B, 128]
        zz = torch.relu(self.fc_z(z))         # [B, 128]
        fused = torch.relu(self.fc(torch.cat([hz, zz], dim=1)))  # [B, 128]

        dec_in = fused[:, :, None].repeat(1, 1, self.hist_len)   # [B,128,H]
        out = self.dec(dec_in)                                   # [B,2,~2H]

        target_len = 2 * self.hist_len
        if out.size(-1) < target_len:
            out = F.pad(out, (0, target_len - out.size(-1)))
        elif out.size(-1) > target_len:
            out = out[..., :target_len]

        B, C, L = out.shape
        res = out.reshape(B * C, L)
        res = self.proj_len(res)                                 # (B*2, T)
        res = res.reshape(B, C, self.seq_len)                    # [B,2,T]

        # Bound residuals with tanh and learnable scale
        return torch.tanh(res) * self.res_scale                  # Δy

class Critic(nn.Module):
    """
    WGAN critic; conditions on history by concatenation along channels.
      y : [B, 2, T]  (the FINAL series passed to critic, i.e., baseline + residual)
      h : [B, 2, H]
      ch: [B, 4, H] or empty
      cf: [B, 4, T] or empty
    """
    def __init__(self, hist_len=48, seq_len=48, use_calendar=False):
        super().__init__()
        self.use_calendar = use_calendar
        in_ch_hist = 2 + (4 if use_calendar else 0)
        in_ch_fut  = 2 + (4 if use_calendar else 0)
        self.enc_h = ResTCN(c_in=in_ch_hist, channels=(64, 128))
        self.enc_y = ResTCN(c_in=in_ch_fut,  channels=(64, 128))
        self.fc = nn.Sequential(
            nn.Linear(self.enc_h.out_channels + self.enc_y.out_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, y, h, ch=None, cf=None):
        h_in = torch.cat([h, ch], dim=1) if (self.use_calendar and ch is not None and ch.size(1) > 0) else h
        y_in = torch.cat([y, cf], dim=1) if (self.use_calendar and cf is not None and cf.size(1) > 0) else y
        fh = self.enc_h(h_in).mean(dim=-1)
        fy = self.enc_y(y_in).mean(dim=-1)
        return self.fc(torch.cat([fh, fy], dim=1))
