import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEDEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 300,
                 channels=(64, 128, 256),
                 kernels=(5, 5, 3),
                 pools=(2, 2, 2),
                 dropout: float = 0.30,
                 l2_normalize: bool = False):
        super().__init__()
        assert len(channels) == len(kernels) == len(pools), "channels/kernels/pools length mismatch"
        self.l2_normalize = l2_normalize

        layers = []
        in_ch = 1
        for ch, k, p in zip(channels, kernels, pools):
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=k, padding=k//2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            if p and p > 1:
                layers.append(nn.MaxPool1d(kernel_size=p, stride=p))
            in_ch = ch
        self.conv = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        proj_in = in_ch * 2
        self.proj = nn.Sequential(
            nn.Linear(proj_in, proj_in * 2), nn.GELU(), nn.LayerNorm(proj_in * 2),
            nn.Dropout(dropout), nn.Linear(proj_in * 2, embedding_dim)
        )

    def forward(self, x):  
        x = x.unsqueeze(1).float()        
        h = self.conv(x)                
        h_avg = self.gap(h).squeeze(-1)    
        h_max = self.gmp(h).squeeze(-1)  
        h_cat = torch.cat([h_avg, h_max], dim=1)  
        z = self.proj(h_cat)               
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z


class DistanceCalib(nn.Module):
    def __init__(self, init_scale=1.0, init_bias=0.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale + 1e-6)))
        self.bias      = nn.Parameter(torch.tensor(init_bias))
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
def bench_timing(encoder: nn.Module, T: int, runs: int = 200, device: torch.device = torch.device("cpu")):
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

def get_last_conv_channels(model: CNNEDEncoder):
    ch = None
    for m in model.conv.modules():
        if isinstance(m, nn.Conv1d):
            ch = m.out_channels
    return ch
