import os
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import load_triplet_datasets, train_test_split
from rebind.baselines.cgk import CGKEnsemble, fit_linear_calibrator, apply_linear


def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pearsonr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float((a @ b) / den)

def _rankdata(x):
    x = np.asarray(x); order = x.argsort(kind="mergesort")
    r = np.empty_like(order, dtype=float); r[order] = np.arange(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and x[order[j+1]] == x[order[i]]:
            j += 1
        if j > i:
            avg = (i + j)/2.0
            r[i:j+1] = avg
        i = j + 1
    return r

def spearmanr(a, b): return pearsonr(_rankdata(a), _rankdata(b))
def rmse(p, t): p, t = np.asarray(p, float), np.asarray(t, float); return float(np.sqrt(np.mean((p-t)**2)))
def mae(p, t):  p, t = np.asarray(p, float), np.asarray(t, float); return float(np.mean(np.abs(p-t)))

def pair_metrics(pred_all, true_all):
    return dict(RMSE=rmse(pred_all, true_all),
                MAE=mae(pred_all, true_all),
                Pearson_r=pearsonr(pred_all, true_all),
                Spearman_rho=spearmanr(pred_all, true_all))

def ranking_metrics(pred_ap, pred_an, margin=0.0):
    ap, an = np.asarray(pred_ap, float), np.asarray(pred_an, float)
    acc = float(np.mean(ap + margin < an))
    return dict(TripletAcc=acc, ViolationRate=1.0-acc, AvgMargin=float(np.mean(an-ap)))

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p


def collect_with_timing(cgk: CGKEnsemble, loader, L_base, max_batches=None, low_edit_k=None,
                        agg="median", mode="ignore_tail"):
    raw_ap_all, raw_an_all, raw_pn_all = [], [], []
    true_ap_all, true_an_all, true_pn_all = [], [], []
    total_time = 0.0; total_pairs = 0; batches = 0

    for a, p, n, dap, dan, dpn in tqdm(loader, desc="[CGK collect]"):
        A = a.numpy().astype(np.uint8)[:, :L_base]
        P = p.numpy().astype(np.uint8)[:, :L_base]
        N = n.numpy().astype(np.uint8)[:, :L_base]
        dap_np, dan_np, dpn_np = dap.numpy(), dan.numpy(), dpn.numpy()

        if low_edit_k is not None:
            mask = (dap_np <= low_edit_k) & (dan_np <= low_edit_k) & (dpn_np <= low_edit_k)
            if not np.any(mask):
                batches += 1
                if max_batches is not None and batches >= max_batches: break
                continue
            A, P, N = A[mask], P[mask], N[mask]
            dap_np, dan_np, dpn_np = dap_np[mask], dan_np[mask], dpn_np[mask]
            if A.shape[0] == 0:
                batches += 1
                if max_batches is not None and batches >= max_batches: break
                continue

        t0 = time.perf_counter()
        dap_hat, dan_hat, dpn_hat = cgk.embed_and_dist_triplet(A, P, N, agg=agg, mode=mode)
        t1 = time.perf_counter()

        total_time += (t1 - t0)
        B = len(dap_np)
        total_pairs += 3 * B

        raw_ap_all.append(dap_hat); raw_an_all.append(dan_hat); raw_pn_all.append(dpn_hat)
        true_ap_all.append(dap_np);  true_an_all.append(dan_np);  true_pn_all.append(dpn_np)

        batches += 1
        if max_batches is not None and batches >= max_batches: break

    raw_ap = np.concatenate(raw_ap_all) if raw_ap_all else np.array([])
    raw_an = np.concatenate(raw_an_all) if raw_an_all else np.array([])
    raw_pn = np.concatenate(raw_pn_all) if raw_pn_all else np.array([])
    true_ap = np.concatenate(true_ap_all) if true_ap_all else np.array([])
    true_an = np.concatenate(true_an_all) if true_an_all else np.array([])
    true_pn = np.concatenate(true_pn_all) if true_pn_all else np.array([])

    timing = dict(
        enc_ms_per_pair  = (total_time / max(total_pairs,1)) * 1e3,
        enc_ms_per_batch = (total_time / max(batches,1)) * 1e3,
        n_pairs_eval     = int(total_pairs),
        n_batches_eval   = int(batches),
        backend          = "numpy"
    )
    return raw_ap, raw_an, raw_pn, true_ap, true_an, true_pn, timing

def collect_rows_for_csv(cgk: CGKEnsemble, loader, L_base, max_batches=None, low_edit_k=None,
                         agg="median", mode="ignore_tail"):
    rows = []
    for a, p, n, dap, dan, dpn in tqdm(loader, desc="[CGK rows]"):
        A = a.numpy().astype(np.uint8)[:, :L_base]
        P = p.numpy().astype(np.uint8)[:, :L_base]
        N = n.numpy().astype(np.uint8)[:, :L_base]
        dap_np, dan_np, dpn_np = dap.numpy(), dan.numpy(), dpn.numpy()

        if low_edit_k is not None:
            mask = (dap_np <= low_edit_k) & (dan_np <= low_edit_k) & (dpn_np <= low_edit_k)
            if not np.any(mask):
                if max_batches is not None: max_batches -= 1
                continue
            A, P, N = A[mask], P[mask], N[mask]
            dap_np, dan_np, dpn_np = dap_np[mask], dan_np[mask], dpn_np[mask]
            if A.shape[0] == 0:
                if max_batches is not None: max_batches -= 1
                continue

        dap_hat, dan_hat, dpn_hat = cgk.embed_and_dist_triplet(A, P, N, agg=agg, mode=mode)
        for i in range(len(dap_hat)):
            rows.append({"pair":"ap","true_ed":float(dap_np[i]), "est_ed_raw":float(dap_hat[i])})
            rows.append({"pair":"an","true_ed":float(dan_np[i]), "est_ed_raw":float(dan_hat[i])})
            rows.append({"pair":"pn","true_ed":float(dpn_np[i]), "est_ed_raw":float(dpn_hat[i])})

        if max_batches is not None:
            max_batches -= 1
            if max_batches <= 0: break
    return rows


def main():
    ap = argparse.ArgumentParser(description="CGK Ensemble (NumPy only)")
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default="Binary")
    ap.add_argument("--method_name", type=str, default="CGK-Ens")

    ap.add_argument("--K", type=int, default=90)
    ap.add_argument("--L", type=int, default=100)            
    ap.add_argument("--pad_ratio", type=float, default=0.3)
    ap.add_argument("--num_sequences", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--out_scale", type=float, default=6.0)
    ap.add_argument("--agg", type=str, default="median", choices=["median","min","mean"])
    ap.add_argument("--mode", type=str, default="ignore_tail", choices=["count_all","ignore_tail"])

    ap.add_argument("--calib_batches", type=int, default=80)
    ap.add_argument("--eval_batches", type=int, default=200)
    ap.add_argument("--low_edit_k", type=int, default=None)
    ap.add_argument("--triplet_margin", type=float, default=0.0)

    ap.add_argument("--out_dir", type=str, default="./results/cgk_numpy")
    ap.add_argument("--csv_out", type=str, default=None)
    ap.add_argument("--csv_max_points", type=int, default=200000)

    args = ap.parse_args()
    set_seed(args.seed)

    DATASET_FOLDER = args.datasets or os.path.join(args.project_root, "datasets")
    Tpad = int(args.L * (1.0 + args.pad_ratio))

    print(f"[INFO] Loading K={args.K}, num_sequences={args.num_sequences}, Tpad={Tpad}")
    trip_sets = load_triplet_datasets(DATASET_FOLDER, [args.K], args.num_sequences, Tpad)
    train_trip, test_trip = train_test_split(trip_sets)
    train_loader = DataLoader(train_trip[0], batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_trip[0],  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    cgk = CGKEnsemble(L=args.L, m=args.m, out_scale=args.out_scale, seed=args.seed)
    print(f"[CGK] L={args.L}, M={cgk.M}, m={args.m}, agg={args.agg}, mode={args.mode}")

    raw_ap_tr, raw_an_tr, raw_pn_tr, true_ap_tr, true_an_tr, true_pn_tr, _ = collect_with_timing(
        cgk, train_loader, L_base=args.L, max_batches=args.calib_batches,
        low_edit_k=args.low_edit_k, agg=args.agg, mode=args.mode
    )
    a_lin, b_lin = fit_linear_calibrator(
        np.concatenate([raw_ap_tr,  raw_an_tr,  raw_pn_tr]),
        np.concatenate([true_ap_tr, true_an_tr, true_pn_tr])
    )
    print(f"[Calibrator] ED ≈ {a_lin:.6f} * H + {b_lin:.6f}")

    raw_ap, raw_an, raw_pn, true_ap, true_an, true_pn, timing = collect_with_timing(
        cgk, test_loader, L_base=args.L, max_batches=args.eval_batches,
        low_edit_k=args.low_edit_k, agg=args.agg, mode=args.mode
    )
    pred_ap = apply_linear(raw_ap, a_lin, b_lin)
    pred_an = apply_linear(raw_an, a_lin, b_lin)
    pred_pn = apply_linear(raw_pn, a_lin, b_lin)

    pred_all = np.concatenate([pred_ap, pred_an, pred_pn])
    true_all = np.concatenate([true_ap, true_an, true_pn])

    s1 = pair_metrics(pred_all, true_all)
    s2 = ranking_metrics(pred_ap, pred_an, margin=args.triplet_margin)

    print("\n=== Fidelity ===");  [print(f"{k:>12}: {v:.6f}") for k,v in s1.items()]
    print("\n=== Ranking  ===");  [print(f"{k:>12}: {v:.6f}") for k,v in s2.items()]
    print("\n=== Timing   ===");  [print(f"{k:>16}: {v:.3f}") if "ms" in k else print(f"{k:>16}: {v}") for k,v in timing.items()]

    out_dir = ensure_dir(args.out_dir)
    pd.DataFrame([s1], index=[args.method_name]).to_csv(os.path.join(out_dir, "fidelity_metrics.csv"))
    pd.DataFrame([s2], index=[args.method_name]).to_csv(os.path.join(out_dir, "ranking_metrics.csv"))

    enc_params_M = (args.m * cgk.M * 2) / 1e6  
    point = {
        "dataset": args.dataset_name, "method": args.method_name,
        "enc_ms_per_pair": timing["enc_ms_per_pair"], "enc_ms_per_batch": timing["enc_ms_per_batch"],
        "n_pairs_eval": timing["n_pairs_eval"],
        "RMSE": s1["RMSE"], "MAE": s1["MAE"], "Pearson_r": s1["Pearson_r"], "Spearman_rho": s1["Spearman_rho"],
        "TripletAcc": s2["TripletAcc"], "ViolationRate": s2["ViolationRate"], "AvgMargin": s2["AvgMargin"],
        "enc_params_M": enc_params_M, "calib_params_M": 0.0, "backend": timing["backend"]
    }
    pd.DataFrame([point]).to_csv(os.path.join(out_dir, "fidelity_point.csv"), index=False)

    rows = collect_rows_for_csv(
        cgk, test_loader, L_base=args.L, max_batches=args.eval_batches,
        low_edit_k=args.low_edit_k, agg=args.agg, mode=args.mode
    )
    csv_path = args.csv_out or os.path.join(out_dir, f"ed_scatter_{args.dataset_name}_{args.method_name}_K{args.K}_L{args.L}_m{args.m}_M{cgk.M}.csv")
    if rows:
        if args.csv_max_points is not None and len(rows) > args.csv_max_points:
            idx = np.random.RandomState(args.seed).choice(len(rows), size=args.csv_max_points, replace=False)
            rows = [rows[i] for i in idx]
        with open(csv_path, "w") as f:
            f.write("dataset,method,pair,true_ed,est_ed_raw,est_ed\n")
            for r in rows:
                est = a_lin * r["est_ed_raw"] + b_lin
                f.write(f"{args.dataset_name},{args.method_name},{r['pair']},{r['true_ed']},{r['est_ed_raw']},{est}\n")

    summary = {
        "model": "CGK-Ens (NumPy)",
        "config": {"L": args.L, "M": cgk.M, "m": args.m, "agg": args.agg, "mode": args.mode},
        "fidelity": s1, "ranking": s2, "timing": timing,
        "params_M": {"encoder_like": enc_params_M, "calibrator": 0.0}
    }
    with open(os.path.join(out_dir, "cgk_numpy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {out_dir}")
    print(" - fidelity_metrics.csv")
    print(" - ranking_metrics.csv")
    print(" - fidelity_point.csv")
    print(f" - scatter CSV: {csv_path}")
    print(" - cgk_numpy_summary.json")

if __name__ == "__main__":
    main()

