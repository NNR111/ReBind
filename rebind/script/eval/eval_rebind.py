import os
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_triplet_datasets, load_small_datasets, train_test_split
from rebind.models.rebind_model import BiGRUEncoder, DistanceCalib, MultiStageBidirectionalGRUDecoder

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))

def pearsonr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    a = a - a.mean(); b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom > 0 else 0.0

def _rankdata(x):
    tmp = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(tmp, dtype=float)
    ranks[tmp] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    avg = sums / counts
    return avg[inv] + 1.0 

def spearmanr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ra = _rankdata(a); rb = _rankdata(b)
    return pearsonr(ra, rb)


def load_encoder_and_calib(ckpt_path, device, fallback_cfg):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {})
    enc_dim     = cfg.get("embedding_dim", fallback_cfg["embedding_dim"])
    enc_hidden  = cfg.get("enc_hidden_dim", fallback_cfg["hidden_dim"])
    enc_layers  = cfg.get("enc_num_layers", fallback_cfg["num_layers"])
    enc_drop    = cfg.get("enc_dropout",   fallback_cfg["dropout"])
    enc_l2_norm = cfg.get("enc_l2_norm",   fallback_cfg["l2_normalize"])

    encoder = BiGRUEncoder(embedding_dim=enc_dim,
                           hidden_size=enc_hidden,
                           num_layers=enc_layers,
                           dropout=enc_drop,
                           l2_normalize=enc_l2_norm).to(device)
    encoder.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    calib = DistanceCalib().to(device)
    if "calib" in ckpt:
        calib.load_state_dict(ckpt["calib"], strict=False)
    calib.eval()
    for p in calib.parameters():
        p.requires_grad = False

    return encoder, calib, enc_dim

@torch.no_grad()
def predict_pairwise_distances(encoder, calib, a, b, device, use_squared=False):
    a = a.to(device).float()
    b = b.to(device).float()
    za = encoder(a); zb = encoder(b)
    if use_squared:
        raw = torch.sum((za - zb) ** 2, dim=1)
    else:
        raw = torch.norm(za - zb, p=2, dim=1)
    return calib(raw).detach().cpu().numpy()


def task_fidelity_and_ranking(args, device):
    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    datasets   = load_triplet_datasets(os.path.join(args.project_root, "datasets"),
                                       [args.K], args.num_sequences, padded_len)
    _, test_sets = train_test_split(datasets)   
    loader = DataLoader(test_sets[0], batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)


    fallback_cfg = dict(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, l2_normalize=args.l2_normalize)
    enc, calib, _ = load_encoder_and_calib(args.encoder_ckpt, device, fallback_cfg)

    preds, gts = [], []
    n_total, n_correct = 0, 0

    for a, p, n, d_ap, d_an, d_pn in tqdm(loader, desc="Eval (fidelity+ranking)"):
        pred_ap = predict_pairwise_distances(enc, calib, a, p, device, args.use_squared_dist)
        pred_an = predict_pairwise_distances(enc, calib, a, n, device, args.use_squared_dist)
        pred_pn = predict_pairwise_distances(enc, calib, p, n, device, args.use_squared_dist)

        preds.extend(np.concatenate([pred_ap, pred_an, pred_pn], axis=0))
        gts.extend(np.concatenate([d_ap.numpy(), d_an.numpy(), d_pn.numpy()], axis=0))

        n_total += len(pred_ap)
        n_correct += int(np.sum(pred_ap < pred_an))

    preds = np.asarray(preds); gts = np.asarray(gts)
    out = {
        "rmse": rmse(preds, gts),
        "mae": mae(preds, gts),
        "pearson": pearsonr(preds, gts),
        "spearman": spearmanr(preds, gts),
        "triplet_accuracy": float(n_correct) / max(1, n_total)
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"eval_fidelity_ranking_K{args.K}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


    if args.dump_scatter:
        idx = np.linspace(0, len(preds)-1, num=min(args.scatter_points, len(preds)), dtype=int)
        csv_path = os.path.join(args.save_dir, f"scatter_K{args.K}.csv")
        with open(csv_path, "w") as f:
            f.write("true_ed,est_ed\n")
            for i in idx:
                f.write(f"{gts[i]},{preds[i]}\n")
        print(f"Scatter CSV saved to {csv_path}")

def task_invertibility(args, device):
    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    datasets   = load_small_datasets(os.path.join(args.project_root, "datasets"),
                                     [args.K], padded_len)
    _, test_sets = train_test_split(datasets)
    loader = DataLoader(test_sets[0], batch_size=args.batch_size, shuffle=False)


    fallback_cfg = dict(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, l2_normalize=args.l2_normalize)
    enc, _, enc_dim = load_encoder_and_calib(args.encoder_ckpt, device, fallback_cfg)


    dec_path = args.decoder_ckpt if os.path.isabs(args.decoder_ckpt) \
        else os.path.join(args.project_root, args.decoder_ckpt)
    decoder = MultiStageBidirectionalGRUDecoder(
        embedding_dim=enc_dim,
        hidden_dim=args.dec_hidden_dim,
        output_dim=padded_len,
        num_layers=args.dec_num_layers,
        dropout=args.dec_dropout
    ).to(device)
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    decoder.eval()

    @torch.no_grad()
    def batch_hamming(pred, target):
        pred = pred.cpu().numpy().astype(np.int32)
        target = target.cpu().numpy().astype(np.int32)
        return float(np.mean(pred != target))

    total_h = 0.0
    total_n = 0
    with torch.no_grad():
        for original, _, _ in tqdm(loader, desc="Eval (invertibility)"):
            original = original.to(device).float()
            z = enc(original)
            out = decoder(z)           # probs
            pred = (out > 0.5).float()
            total_h += batch_hamming(pred, original) * original.size(0)
            total_n += original.size(0)

    avg_hamming = total_h / max(1, total_n)
    out = {"hamming_loss": avg_hamming}
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"eval_invertibility_K{args.K}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate ReBind models")
    ap.add_argument("--project_root", type=str, default="/home/coder/my code/invertible_separate")
    ap.add_argument("--save_dir", type=str, default="./trained_models")
    ap.add_argument("--K", type=int, default=90)
    ap.add_argument("--num_sequences", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--padding_ratio", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=256)


    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--l2_normalize", action="store_true", default=False)
    ap.add_argument("--use_squared_dist", action="store_true", default=False)


    ap.add_argument("--encoder_ckpt", type=str,
                    default="trained_models/final_model_BiGRUEncoder_K90_calibrated_ours.pth")
    ap.add_argument("--decoder_ckpt", type=str,
                    default="trained_models/decoder_final_512_k90.pth")


    ap.add_argument("--task", type=str, default="all",
                    choices=["all", "fidelity", "ranking", "invertibility", "fidelity+ranking"])


    ap.add_argument("--dump_scatter", action="store_true", default=False)
    ap.add_argument("--scatter_points", type=int, default=5000)


    ap.add_argument("--dec_hidden_dim", type=int, default=512)
    ap.add_argument("--dec_num_layers", type=int, default=2)
    ap.add_argument("--dec_dropout", type=float, default=0.30)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task in ["fidelity", "ranking", "fidelity+ranking", "all"]:
        task_fidelity_and_ranking(args, device)

    if args.task in ["invertibility", "all"]:
        task_invertibility(args, device)


# # 1) Fidelity + Ranking (triplets)
# python scripts/eval.py --task fidelity+ranking \
#   --project_root "/home/coder/my code/invertible_separate" \
#   --encoder_ckpt trained_models/final_model_BiGRUEncoder_K90_calibrated_ours.pth \
#   --dump_scatter

# # 2) Invertibility (originals only, Hamming loss)
# python scripts/eval.py --task invertibility \
#   --project_root "/home/coder/my code/invertible_separate" \
#   --encoder_ckpt trained_models/final_model_BiGRUEncoder_K90_calibrated_ours.pth \
#   --decoder_ckpt trained_models/decoder_final_512_k90.pth

# # 3) Everything
# python scripts/eval.py --task all --project_root "/home/coder/my code/invertible_separate"
