
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  
    def forward(self, x):  
        return x + self.pe[:, :x.size(1), :]


class TransformerEDEncoder(nn.Module):
    def __init__(self, embedding_dim=300, model_dim=256, nhead=8, num_layers=4,
                 dropout=0.1, l2_normalize=True):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.in_proj = nn.Linear(1, model_dim)
        self.in_norm = nn.LayerNorm(model_dim)
        self.posenc = PositionalEncoding(model_dim, max_len=4096)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=4*model_dim,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(model_dim))

        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(model_dim, 2*model_dim), nn.GELU(), nn.LayerNorm(2*model_dim),
            nn.Dropout(dropout), nn.Linear(2*model_dim, embedding_dim)
        )

    def forward(self, x): 
        b, t = x.shape
        h = self.in_proj(x.unsqueeze(-1).float()) 
        h = self.in_norm(h)
        h = self.posenc(h)
        cls = self.cls.expand(b, 1, -1)            
        h = torch.cat([cls, h], dim=1)            
        h = self.encoder(self.drop(h))
        z = self.proj(h[:, 0, :])                  
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

def cosine_distance(z1, z2):
    return 1.0 - (z1 * z2).sum(dim=1).clamp(-1.0, 1.0)

def vicreg_regularizer(z, eps: float = 1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std))

    B, D = z.shape
    cov = (z.T @ z) / max(1, (B - 1))
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off**2).sum() / D
    return var_loss, cov_loss

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
        _ = encoder(x); maybe_sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); _ = encoder(x); maybe_sync(); t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    t = np.asarray(times)
    return {"enc_ms_median": float(np.median(t)), "enc_ms_mean": float(np.mean(t)), "enc_ms_std": float(np.std(t))}

@torch.no_grad()
def bench_timing_pair_with_calib(encoder: nn.Module, calib: nn.Module, T: int, runs: int = 200, device: torch.device = torch.device("cpu")):
    encoder.eval(); calib.eval()
    a = (torch.rand(1, T, device=device) > 0.5).float()
    b = (torch.rand(1, T, device=device) > 0.5).float()
    for _ in range(20):
        za = encoder(a); zb = encoder(b); d = cosine_distance(za, zb); _ = calib(d); maybe_sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        za = encoder(a); zb = encoder(b)
        d = cosine_distance(za, zb); _ = calib(d)
        maybe_sync(); t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    t = np.asarray(times)
    return {"pair_ms_median": float(np.median(t)), "pair_ms_mean": float(np.mean(t)), "pair_ms_std": float(np.std(t))}
