import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    def __init__(self, in_dim, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or in_dim
        self.w = nn.Linear(in_dim, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x): 
        s = self.v(torch.tanh(self.w(x))).squeeze(-1)  
        s = torch.softmax(s, dim=-1)
        s = self.drop(s)
        return torch.bmm(s.unsqueeze(1), x).squeeze(1)  


class BiGRUEncoder(nn.Module):
    def __init__(self,
                 embedding_dim=300,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.30,
                 l2_normalize=False):
        super().__init__()
        self.l2_normalize = l2_normalize

        self.bigru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout
        )

        self.attn = AttnPool(in_dim=2*hidden_size,
                             hidden=2*hidden_size,
                             dropout=dropout)

        proj_in = 4 * hidden_size
        self.proj = nn.Sequential(
            nn.Linear(proj_in, 2*proj_in),
            nn.GELU(),
            nn.LayerNorm(2*proj_in),
            nn.Dropout(dropout),
            nn.Linear(2*proj_in, embedding_dim)
        )

    def forward(self, x): 
        x = x.unsqueeze(-1).float()          
        out, h_n = self.bigru(x)           
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  
        h_attn = self.attn(out)                                   
        h = torch.cat([h_last, h_attn], dim=-1)                   
        z = self.proj(h)                                          
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z

class DistanceCalib(nn.Module):
    def __init__(self, init_scale=1.0, init_bias=0.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale + 1e-6)))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, d):
        return F.softplus(self.log_scale) * d + self.bias


class MultiStageBidirectionalGRUDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.30):
        super().__init__()
        self.output_dim = output_dim

        self.expand_z = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.bigru1 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.bigru2 = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.expand_z(z)                           
        z_seq = z.unsqueeze(1).repeat(1, self.output_dim, 1)  
        out, _ = self.bigru1(z_seq)
        out, _ = self.bigru2(out)
        return self.output_head(out).squeeze(-1)       
