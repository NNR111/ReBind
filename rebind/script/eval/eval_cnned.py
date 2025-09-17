import os
import json
import argparse
import math
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_triplet_datasets, train_test_split
from rebind.baselines.cnned import CNNEDEncoder, DistanceCalib, bench_timing, count_params


def rmse(a, b): a=np.asarray(a); b=np.asarray(b); return float(np.sqrt(np.mean((a-b)**2)))
def mae(a, b):  a=np.asarray(a); b=np.asarray(b); return float(np.mean(np.abs(a-b)))
def pearsonr(a, b):
    a=np.asarray(a); b=np.asarray(b)
    a=a-a.mean(); b=b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/den) if den>0 else 0.0
def _rankdata(x):
    x=np.asarray(x); idx=np.argsort(x, kind="mergesort")
    r = np.empty_like(idx, dtype=float); r[idx]=np.arange(len(x), dtype=float)
    i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[idx[j+1]]==x[idx[i]]: j+=1
        if j>i: r[i:j+1]=(i+j)/2.0
        i=j+1
    return r+1.0
def spearmanr(a,b): return pearsonr(_rankdata(a), _rankdata(b))

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

@torch.no_grad()
def eval_fidelity(encoder, calib, loader, device):
    encoder.eval(); calib.eval()
    gts, preds = [], []
    for a, p, n, d_ap, d_an, d_pn in loader:
        a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
        d_ap=d_ap.to(device).float().squeeze(); d_an=d_an.to(device).float().squeeze(); d_pn=d_pn.to(device).float().squeeze()
        z_a=encoder(a); z_p=encoder(p); z_n=encoder(n)
        r_ap=torch.norm(z_a-z_p, dim=1); r_an=torch.norm(z_a-z_n, dim=1); r_pn=torch.norm(z_p-z_n, dim=1)
        p_ap=calib(r_ap); p_an=calib(r_an); p_pn=calib(r_pn)
        gts.append(torch.stack([d_ap,d_an,d_pn],1).cpu().numpy())
        preds.append(torch.stack([p_ap,p_an,p_pn],1).cpu().numpy())
    gts=np.concatenate(gts,0).reshape(-1); preds=np.concatenate(preds,0).reshape(-1)
    return {
        "rmse": rmse(gts,preds),
        "mae": mae(gts,preds),
        "pearson": pearsonr(gts,preds),
        "spearman": spearmanr(gts,preds),
        "num_pairs": int(gts.size),
    }, gts, preds

@torch.no_grad()
def eval_ranking(encoder, calib, loader, device):
    encoder.eval(); calib.eval()
    ok_ap_an=0; ok_pn_an=0; total=0
    for a, p, n, *_ in loader:
        a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
        z_a=encoder(a); z_p=encoder(p); z_n=encoder(n)
        r_ap=torch.norm(z_a-z_p, dim=1); r_an=torch.norm(z_a-z_n, dim=1); r_pn=torch.norm(z_p-z_n, dim=1)
        p_ap=calib(r_ap); p_an=calib(r_an); p_pn=calib(r_pn)
        ok_ap_an += torch.sum(p_ap < p_an).item()
        ok_pn_an += torch.sum(p_pn < p_an).item()
        total += a.size(0)
    return {
        "triplet_acc_ap_an": ok_ap_an/total,
        "triplet_acc_pn_an": ok_pn_an/total,
        "triplet_acc_mean": (ok_ap_an+ok_pn_an)/(2.0*total),
        "num_triplets": int(total),
    }

def save_plots(ed_true, d_pred, figs_dir):
    plt.figure(figsize=(5.2,4.0))
    plt.scatter(ed_true, d_pred, s=6, alpha=0.35)
    lo=min(ed_true.min(), d_pred.min()); hi=max(ed_true.max(), d_pred.max())
    plt.plot([lo,hi],[lo,hi], linewidth=1.0)
    plt.xlabel("True ED"); plt.ylabel("Predicted (calibrated)")
    plt.title("ED vs Predicted (CNN-ED)")
    plt.grid(True, linestyle=":", linewidth=0.6); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"cnned_ed_vs_pred.png"), dpi=180); plt.close()

    resid = d_pred - ed_true
    plt.figure(figsize=(5.2,3.8))
    plt.hist(resid, bins=50)
    plt.xlabel("Residual (pred - true)"); plt.ylabel("Count")
    plt.title("Residuals (CNN-ED)")
    plt.grid(True, linestyle=":", linewidth=0.6); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"cnned_residual_hist.png"), dpi=180); plt.close()

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    dataset_folder = args.dataset_folder or os.path.join(args.project_root, "datasets")
    out_dir = ensure_dir(args.out_dir)
    figs_dir = ensure_dir(os.path.join(out_dir, "figs"))


    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    emb_dim = cfg.get("embedding_dim", args.embedding_dim)
    channels = tuple(int(x) for x in cfg.get("channels", args.channels).split(",") if x.strip())
    kernels  = tuple(int(x) for x in cfg.get("kernels",  args.kernels).split(",") if x.strip())
    pools    = tuple(int(x) for x in cfg.get("pools",    args.pools).split(",") if x.strip())

    encoder = CNNEDEncoder(
        embedding_dim=emb_dim, channels=channels, kernels=kernels, pools=pools,
        dropout=args.dropout, l2_normalize=args.l2_normalize
    ).to(device)
    enc_state = ckpt["encoder"] if "encoder" in ckpt else ckpt
    encoder.load_state_dict(enc_state, strict=False); encoder.eval()

    calib = DistanceCalib().to(device)
    if "calib" in ckpt: calib.load_state_dict(ckpt["calib"], strict=False)
    calib.eval()


    datasets = load_triplet_datasets(dataset_folder, [args.k], args.num_eval_triplets, padded_len)
    _, test_sets = train_test_split(datasets)
    loader = DataLoader(test_sets[0], batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)


    fidelity, ed_true, d_pred = eval_fidelity(encoder, calib, loader, device)
    print("[Fidelity]", json.dumps(fidelity, indent=2))


    ranking = eval_ranking(encoder, calib, loader, device)
    print("[Ranking]", json.dumps(ranking, indent=2))


    timing = bench_timing(encoder, T=padded_len, runs=200, device=device)
    print("[Timing]", json.dumps(timing, indent=2))


    if args.save_plots:
        save_plots(ed_true, d_pred, figs_dir)


    summary = {
        "model": "CNN-ED",
        "k": args.k,
        "seq_len": args.seq_len,
        "padding_ratio": args.padding_ratio,
        "params_m": count_params(encoder)/1e6,
        "fidelity": fidelity,
        "ranking": ranking,
        "invertibility": {"hamming": None},
        "timing": timing,
    }
    sum_path = os.path.join(out_dir, "cnned_eval_summary.json")
    with open(sum_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"[Saved] {sum_path}")

   
    if args.csv_out:
        row = {
            "model": "CNN-ED",
            "variant": f"C{','.join(map(str,channels))}-K{','.join(map(str,kernels))}-P{','.join(map(str,pools))}-D{emb_dim}",
            "inference_ms": timing.get("enc_ms_median", ""),
            "rmse": fidelity.get("rmse", ""),
            "mse": (fidelity.get("rmse", 0.0)**2) if isinstance(fidelity.get("rmse", None), (float,int)) else "",
            "mae": fidelity.get("mae", ""),
            "pearson": fidelity.get("pearson", ""),
            "spearman": fidelity.get("spearman", ""),
            "triplet_acc": ranking.get("triplet_acc_mean", ""),
            "hamming": "",  # N/A
            "params_m": summary["params_m"],
            "flops_g": ""   # optional
        }
        header = list(row.keys())
        csv_path = os.path.abspath(args.csv_out)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=header)
            if write_header: w.writeheader()
            w.writerow(row)
        print(f"[CSV] appended → {csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--dataset_folder", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)


    ap.add_argument("--k", type=int, default=90)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--padding_ratio", type=float, default=0.3)
    ap.add_argument("--num_eval_triplets", type=int, default=50000)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)


    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--channels", type=str, default="64,128,256")
    ap.add_argument("--kernels",  type=str, default="5,5,3")
    ap.add_argument("--pools",    type=str, default="2,2,2")
    ap.add_argument("--dropout",  type=float, default=0.30)
    ap.add_argument("--l2_normalize", action="store_true", default=False)


    ap.add_argument("--out_dir", type=str, default="./results/cnned")
    ap.add_argument("--csv_out", type=str, default="")
    ap.add_argument("--save_plots", action="store_true", default=True)

    args = ap.parse_args()
    main(args)

