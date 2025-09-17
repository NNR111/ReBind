import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_triplet_datasets, train_test_split
from rebind.baselines.gru import GRUEncoder, DistanceCalib, compute_dist, count_params

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    dataset_folder = args.dataset_folder or os.path.join(args.project_root, "datasets")
    save_dir = ensure_dir(args.save_dir)


    datasets = load_triplet_datasets(dataset_folder, [args.k], args.num_sequences, padded_len)
    train_sets, test_sets = train_test_split(datasets)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_sets[0], batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(test_sets[0],  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)


    encoder = GRUEncoder(
        embedding_dim=args.embedding_dim, hidden=args.hidden, num_layers=args.num_layers,
        dropout=args.dropout, bidirectional=args.bidirectional, l2_normalize=args.l2_normalize
    ).to(device)
    calib = DistanceCalib().to(device)
    print(f"[GRU] params = {count_params(encoder)/1e6:.3f} M")


    params = list(encoder.parameters()) + list(calib.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    mse = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        encoder.train(); calib.train()
        total = 0.0
        for a, p, n, d_ap, d_an, d_pn in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
            d_ap=d_ap.to(device).squeeze().float()
            d_an=d_an.to(device).squeeze().float()
            d_pn=d_pn.to(device).squeeze().float()

            za, zp, zn = encoder(a), encoder(p), encoder(n)
            rap = compute_dist(za, zp, squared=args.use_squared_dist)
            ran = compute_dist(za, zn, squared=args.use_squared_dist)
            rpn = compute_dist(zp, zn, squared=args.use_squared_dist)

            pap = calib(rap); pan = calib(ran); ppn = calib(rpn)


            loss_reg = (mse(pap, d_ap) + mse(pan, d_an) + mse(ppn, d_pn)) / 3.0

            trip1 = torch.relu(pap - pan + args.triplet_margin)
            trip2 = torch.relu(ppn - pan + args.triplet_margin)
            loss_trip = (trip1 + trip2).mean()

            loss = loss_reg + args.lambda_triplet * loss_trip

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            total += float(loss.item())

        avg_train = total / max(1, len(train_loader))
        train_losses.append(avg_train)
        print(f" epoch {epoch} | train loss {avg_train:.4f}")

        # Validation
        encoder.eval(); calib.eval()
        with torch.no_grad():
            total_val = 0.0
            for a, p, n, d_ap, d_an, d_pn in val_loader:
                a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
                d_ap=d_ap.to(device).squeeze().float()
                d_an=d_an.to(device).squeeze().float()
                d_pn=d_pn.to(device).squeeze().float()

                za, zp, zn = encoder(a), encoder(p), encoder(n)
                rap = compute_dist(za, zp, squared=args.use_squared_dist)
                ran = compute_dist(za, zn, squared=args.use_squared_dist)
                rpn = compute_dist(zp, zn, squared=args.use_squared_dist)

                pap = calib(rap); pan = calib(ran); ppn = calib(rpn)
                val_loss = (nn.functional.mse_loss(pap, d_ap) +
                            nn.functional.mse_loss(pan, d_an) +
                            nn.functional.mse_loss(ppn, d_pn)) / 3.0
                total_val += float(val_loss.item())

            avg_val = total_val / max(1, len(val_loader))
            val_losses.append(avg_val)
            print(f" val loss {avg_val:.4f}")
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                ckpt_path = os.path.join(save_dir, f"best_model_{args.model_name}.pth")
                torch.save({
                    "encoder": encoder.state_dict(),
                    "calib": calib.state_dict(),
                    "config": {
                        "embedding_dim": args.embedding_dim,
                        "hidden": args.hidden,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "bidirectional": bool(args.bidirectional),
                        "l2_normalize": bool(args.l2_normalize),
                        "use_squared_dist": bool(args.use_squared_dist),
                        "lambda_triplet": args.lambda_triplet,
                        "triplet_margin": args.triplet_margin
                    }
                }, ckpt_path)
                print(f" saved best → {ckpt_path}")


    final_path = os.path.join(save_dir, f"final_model_{args.model_name}.pth")
    torch.save({
        "encoder": encoder.state_dict(),
        "calib": calib.state_dict(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": bool(args.bidirectional),
            "l2_normalize": bool(args.l2_normalize),
        }
    }, final_path)
    print(f" saved final → {final_path}")

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss (calib MSE)")
    plt.title("GRU + Calibrator")
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig_path = os.path.join(save_dir, f"loss_curve_{args.model_name}.png")
    plt.savefig(fig_path, dpi=160)
    print(f" curve → {fig_path}")

    with open(os.path.join(save_dir, f"README_{args.model_name}.json"), "w") as f:
        json.dump({
            "paths": {"best": os.path.join(save_dir, f"best_model_{args.model_name}.pth"),
                      "final": final_path},
            "note": "Checkpoint {'encoder','calib','config'}"
        }, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--dataset_folder", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)


    ap.add_argument("--k", type=int, default=90)
    ap.add_argument("--num_sequences", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--padding_ratio", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)


    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--l2_normalize", action="store_true", default=False)


    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_triplet", type=float, default=0.2)
    ap.add_argument("--triplet_margin", type=float, default=0.1)
    ap.add_argument("--use_squared_dist", action="store_true", default=False)
    ap.add_argument("--grad_clip", type=float, default=1.0)


    ap.add_argument("--save_dir", type=str, default="./trained_models")
    ap.add_argument("--model_name", type=str, default="GRU_K90")
    args = ap.parse_args()
    main(args)

