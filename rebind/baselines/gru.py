import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, embedding_dim=300, hidden=256, num_layers=2,
                 dropout=0.30, bidirectional=True, l2_normalize=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.l2_normalize  = l2_normalize
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(), nn.LayerNorm(out_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, embedding_dim)
        )

    def forward(self, x):  
        x = x.unsqueeze(-1).float()     
        _, h_n = self.gru(x)            
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]                                
        z = self.head(h_last)                               
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z


class DistanceCalib(nn.Module):
    def __init__(self, init_scale=1.0, init_bias=0.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale + 1e-6)))
        self.bias      = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
    def forward(self, d):  
        return F.softplus(self.log_scale) * d + self.bias


def compute_dist(z1, z2, squared: bool = False):
    return torch.sum((z1 - z2) ** 2, dim=1) if squared else torch.norm(z1 - z2, p=2, dim=1)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@torch.no_grad()
def bench_timing_encoder_only(encoder: nn.Module, T: int, runs: int = 200, device: torch.device = torch.device("cpu")):
    encoder.eval()
    x = (torch.rand(1, T, device=device) > 0.5).float()
    for _ in range(20):
        _ = encoder(x)
        maybe_sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = encoder(x)
        maybe_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times = np.asarray(times)
    return {
        "enc_ms_median": float(np.median(times)),
        "enc_ms_mean": float(np.mean(times)),
        "enc_ms_std": float(np.std(times)),
    }

@torch.no_grad()
def bench_timing_pair_with_calib(encoder: nn.Module, calib: nn.Module, T: int, runs: int = 200,
                                 device: torch.device = torch.device("cpu"), squared_dist: bool = False):
    encoder.eval(); calib.eval()
    a = (torch.rand(1, T, device=device) > 0.5).float()
    b = (torch.rand(1, T, device=device) > 0.5).float()
    for _ in range(20):
        za = encoder(a); zb = encoder(b)
        d  = compute_dist(za, zb, squared=squared_dist)
        _  = calib(d)
        maybe_sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        za = encoder(a); zb = encoder(b)
        d  = compute_dist(za, zb, squared=squared_dist)
        _  = calib(d)
        maybe_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times = np.asarray(times)
    return {
        "pair_ms_median": float(np.median(times)),
        "pair_ms_mean": float(np.mean(times)),
        "pair_ms_std": float(np.std(times)),
    }
