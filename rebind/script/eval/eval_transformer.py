import os
import json
import argparse
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_triplet_datasets, train_test_split
from rebind.baselines.transformer import (
    TransformerEDEncoder, DistanceCalib, cosine_distance,
    bench_timing_encoder_only, bench_timing_pair_with_calib, count_params
)


def rmse(a,b): a=np.asarray(a); b=np.asarray(b); return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b):  a=np.asarray(a); b=np.asarray(b); return float(np.mean(np.abs(a-b)))
def pearsonr(a,b):
    a=np.asarray(a); b=np.asarray(b); a=a-a.mean(); b=b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b); return float((a@b)/den) if den>0 else 0.0
def _rankdata(x):
    x=np.asarray(x); idx=np.argsort(x, kind="mergesort")
    r=np.empty_like(idx, dtype=float); r[idx]=np.arange(len(x), dtype=float)
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
        za, zp, zn = encoder(a), encoder(p), encoder(n)
        rap = cosine_distance(za, zp); ran = cosine_distance(za, zn); rpn = cosine_distance(zp, zn)
        pap, pan, ppn = calib(rap), calib(ran), calib(rpn)
        gts.append(torch.stack([d_ap,d_an,d_pn],1).cpu().numpy())
        preds.append(torch.stack([pap,pan,ppn],1).cpu().numpy())
    gts=np.concatenate(gts,0).reshape(-1); preds=np.concatenate(preds,0).reshape(-1)
    return {"rmse": rmse(gts,preds), "mae": mae(gts,preds), "pearson": pearsonr(gts,preds),
            "spearman": spearmanr(gts,preds), "num_pairs": int(gts.size)}, gts, preds

@torch.no_grad()
def eval_ranking(encoder, calib, loader, device):
    encoder.eval(); calib.eval()
    ok_ap_an=0; ok_pn_an=0; total=0
    for a, p, n, *_ in loader:
        a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
        za, zp, zn = encoder(a), encoder(p), encoder(n)
        rap = cosine_distance(za, zp); ran = cosine_distance(za, zn); rpn = cosine_distance(zp, zn)
        pap, pan, ppn = calib(rap), calib(ran), calib(rpn)
        ok_ap_an += torch.sum(pap < pan).item()
        ok_pn_an += torch.sum(ppn < pan).item()
        total += a.size(0)
    return {"triplet_acc_ap_an": ok_ap_an/total, "triplet_acc_pn_an": ok_pn_an/total,
            "triplet_acc_mean": (ok_ap_an+ok_pn_an)/(2.0*total), "num_triplets": int(total)}

def save_plots(ed_true, d_pred, figs_dir, tag="transformer"):
    plt.figure(figsize=(5.2,4.0))
    plt.scatter(ed_true, d_pred, s=6, alpha=0.35)
    lo=min(ed_true.min(), d_pred.min()); hi=max(ed_true.max(), d_pred.max())
    plt.plot([lo,hi],[lo,hi], linewidth=1.0)
    plt.xlabel("True ED"); plt.ylabel("Predicted (calibrated)")
    plt.title("ED vs Predicted (Transformer-ED)")
    plt.grid(True, linestyle=":", linewidth=0.6); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{tag}_ed_vs_pred.png"), dpi=180); plt.close()

    resid = d_pred - ed_true
    plt.figure(figsize=(5.2,3.8))
    plt.hist(resid, bins=50)
    plt.xlabel("Residual (pred - true)"); plt.ylabel("Count")
    plt.title("Residuals (Transformer-ED)")
    plt.grid(True, linestyle=":", linewidth=0.6); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{tag}_residual_hist.png"), dpi=180); plt.close()

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Tpad = int(args.seq_len * (1.0 + args.padding_ratio))
    dataset_folder = args.dataset_folder or os.path.join(args.project_root, "datasets")

    out_dir = ensure_dir(args.out_dir); figs_dir = ensure_dir(os.path.join(out_dir, "figs"))


    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    enc = TransformerEDEncoder(
        embedding_dim=cfg.get("embedding_dim", args.embedding_dim),
        model_dim=cfg.get("model_dim", args.model_dim),
        nhead=cfg.get("nhead", args.nhead),
        num_layers=cfg.get("num_layers", args.num_layers),
        dropout=cfg.get("dropout", args.dropout),
        l2_normalize=True
    ).to(device)
    enc.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt, strict=False)
    enc.eval()

    cal = DistanceCalib().to(device)
    if "calib" in ckpt: cal.load_state_dict(ckpt["calib"], strict=False)
    if "calibrator" in ckpt: cal.load_state_dict(ckpt["calibrator"], strict=False)
    cal.eval()


    datasets = load_triplet_datasets(dataset_folder, [args.k], args.num_eval_triplets, Tpad)
    _, test_sets = train_test_split(datasets)
    loader = DataLoader(test_sets[0], batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"))


    fidelity, ed_true, d_pred = eval_fidelity(enc, cal, loader, device)
    ranking = eval_ranking(enc, cal, loader, device)
    print("[Fidelity]", json.dumps(fidelity, indent=2))
    print("[Ranking]", json.dumps(ranking, indent=2))


    timing_enc  = bench_timing_encoder_only(enc, T=Tpad, runs=200, device=device)
    timing_pair = bench_timing_pair_with_calib(enc, cal, T=Tpad, runs=200, device=device)
    print("[Timing-enc ]", json.dumps(timing_enc, indent=2))
    print("[Timing-pair]", json.dumps(timing_pair, indent=2))


    if args.save_plots: save_plots(ed_true, d_pred, figs_dir, tag="transformer")


    summary = {
        "model": "Transformer-ED",
        "k": args.k, "seq_len": args.seq_len, "padding_ratio": args.padding_ratio,
        "params_m": (count_params(enc) / 1e6),
        "fidelity": fidelity, "ranking": ranking,
        "invertibility": {"hamming": None},
        "timing": {"encoder_only": timing_enc, "pair_with_calib": timing_pair},
        "config_used": {
            "embedding_dim": cfg.get("embedding_dim", args.embedding_dim),
            "model_dim": cfg.get("model_dim", args.model_dim),
            "nhead": cfg.get("nhead", args.nhead),
            "num_layers": cfg.get("num_layers", args.num_layers),
            "dropout": cfg.get("dropout", args.dropout)
        }
    }
    sum_path = os.path.join(out_dir, "transformer_eval_summary.json")
    with open(sum_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"[Saved] {sum_path}")


    if args.csv_out:
        row = {
            "model": "Transformer-ED",
            "variant": f"D{summary['config_used']['model_dim']}-L{summary['config_used']['num_layers']}-H{summary['config_used']['nhead']}",
            "inference_ms": timing_enc.get("enc_ms_median", ""),
            "pair_ms": timing_pair.get("pair_ms_median", ""),
            "rmse": fidelity.get("rmse", ""),
            "mse": (fidelity.get("rmse", 0.0)**2) if isinstance(fidelity.get("rmse", None), (float,int)) else "",
            "mae": fidelity.get("mae", ""),
            "pearson": fidelity.get("pearson", ""),
            "spearman": fidelity.get("spearman", ""),
            "triplet_acc": ranking.get("triplet_acc_mean", ""),
            "hamming": "",  
            "params_m": summary["params_m"],
            "flops_g": ""
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


    pd.DataFrame([{
        "RMSE": fidelity["rmse"], "MAE": fidelity["mae"],
        "Pearson_r": fidelity["pearson"], "Spearman_rho": fidelity["spearman"]
    }], index=["Transformer"]).to_csv(os.path.join(out_dir, "fidelity_metrics.csv"))
    pd.DataFrame([{
        "TripletAcc": ranking["triplet_acc_mean"],
        "TripletAcc_ap_an": ranking["triplet_acc_ap_an"],
        "TripletAcc_pn_an": ranking["triplet_acc_pn_an"]
    }], index=["Transformer"]).to_csv(os.path.join(out_dir, "ranking_metrics.csv"))

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
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)


    ap.add_argument("--out_dir", type=str, default="./results/transformer")
    ap.add_argument("--csv_out", type=str, default="")
    ap.add_argument("--save_plots", action="store_true", default=True)

    args = ap.parse_args()
    main(args)
