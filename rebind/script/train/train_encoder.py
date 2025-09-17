import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_triplet_datasets, train_test_split

from rebind.models.rebind_model import BiGRUEncoder, DistanceCalib
import torch.nn.functional as F

def compute_dist(z1, z2, squared=False):
    return torch.sum((z1 - z2) ** 2, dim=1) if squared else torch.norm(z1 - z2, p=2, dim=1)

def main(args):
    datasets_dir = os.path.join(args.project_root, "datasets")
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    datasets = load_triplet_datasets(datasets_dir, [args.K], args.num_sequences, padded_len)
    train_sets, test_sets = train_test_split(datasets)
    train_loader = DataLoader(train_sets[0], batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_sets[0],  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BiGRUEncoder(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        l2_normalize=args.l2_normalize
    ).to(device)
    calib = DistanceCalib().to(device)

    params = list(encoder.parameters()) + list(calib.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    mse = nn.MSELoss()

    best_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        encoder.train(); calib.train()
        total = 0.0

        for a, p, n, d_ap, d_an, d_pn in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            a = a.to(device).float()
            p = p.to(device).float()
            n = n.to(device).float()
            d_ap = d_ap.to(device).squeeze().float()
            d_an = d_an.to(device).squeeze().float()
            d_pn = d_pn.to(device).squeeze().float()

            z_a = encoder(a); z_p = encoder(p); z_n = encoder(n)
            raw_ap = compute_dist(z_a, z_p, args.use_squared_dist)
            raw_an = compute_dist(z_a, z_n, args.use_squared_dist)
            raw_pn = compute_dist(z_p, z_n, args.use_squared_dist)

            pred_ap = calib(raw_ap); pred_an = calib(raw_an); pred_pn = calib(raw_pn)

            if args.loss_mode.lower() == "relmse":
                eps = 1e-6
                reg = (
                    ((pred_ap - d_ap) ** 2) / ((d_ap + eps) ** 2) +
                    ((pred_an - d_an) ** 2) / ((d_an + eps) ** 2) +
                    ((pred_pn - d_pn) ** 2) / ((d_pn + eps) ** 2)
                ).mean() / 3.0
            else:
                reg = (mse(pred_ap, d_ap) + mse(pred_an, d_an) + mse(pred_pn, d_pn)) / 3.0

            trip1 = F.relu(pred_ap - pred_an + args.triplet_margin)
            trip2 = F.relu(pred_pn - pred_an + args.triplet_margin)
            tri = (trip1 + trip2).mean()

            loss = reg + args.lambda_triplet * tri
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            total += float(loss.item())

        avg_train = total / max(1, len(train_loader))
        train_losses.append(avg_train)
        print(f" Epoch {epoch}: Train Loss = {avg_train:.4f}")

        encoder.eval(); calib.eval()
        with torch.no_grad():
            v = 0.0
            for a, p, n, d_ap, d_an, d_pn in test_loader:
                a = a.to(device).float(); p = p.to(device).float(); n = n.to(device).float()
                d_ap = d_ap.to(device).squeeze().float()
                d_an = d_an.to(device).squeeze().float()
                d_pn = d_pn.to(device).squeeze().float()

                z_a = encoder(a); z_p = encoder(p); z_n = encoder(n)
                raw_ap = compute_dist(z_a, z_p, args.use_squared_dist)
                raw_an = compute_dist(z_a, z_n, args.use_squared_dist)
                raw_pn = compute_dist(z_p, z_n, args.use_squared_dist)

                pred_ap = calib(raw_ap); pred_an = calib(raw_an); pred_pn = calib(raw_pn)

                if args.loss_mode.lower() == "relmse":
                    eps = 1e-6
                    val = (
                        ((pred_ap - d_ap) ** 2) / ((d_ap + eps) ** 2) +
                        ((pred_an - d_an) ** 2) / ((d_an + eps) ** 2) +
                        ((pred_pn - d_pn) ** 2) / ((d_pn + eps) ** 2)
                    ).mean() / 3.0
                else:
                    val = (mse(pred_ap, d_ap) + mse(pred_an, d_an) + mse(pred_pn, d_pn)) / 3.0

                v += float(val.item())

            avg_val = v / max(1, len(test_loader))
            val_losses.append(avg_val)
            print(f" Val Loss: {avg_val:.4f}")

        scheduler.step(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            best_path = os.path.join(save_dir, f"best_model_{args.model_name}.pth")
            torch.save({
                "encoder": encoder.state_dict(),
                "calib": calib.state_dict(),
                "config": {
                    "embedding_dim": args.embedding_dim,
                    "enc_hidden_dim": args.hidden_dim,
                    "enc_num_layers": args.num_layers,
                    "enc_dropout": args.dropout,
                    "enc_l2_norm": args.l2_normalize,
                    "use_squared_dist": args.use_squared_dist,
                    "loss_mode": args.loss_mode,
                    "lambda_triplet": args.lambda_triplet,
                    "triplet_margin": args.triplet_margin
                }
            }, best_path)
            print(f" Best model saved to {best_path}")

    final_path = os.path.join(save_dir, f"final_model_{args.model_name}.pth")
    torch.save({
        "encoder": encoder.state_dict(),
        "calib": calib.state_dict(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "enc_hidden_dim": args.hidden_dim,
            "enc_num_layers": args.num_layers,
            "enc_dropout": args.dropout,
            "enc_l2_norm": args.l2_normalize,
            "use_squared_dist": args.use_squared_dist,
            "loss_mode": args.loss_mode,
            "lambda_triplet": args.lambda_triplet,
            "triplet_margin": args.triplet_margin
        }
    }, final_path)
    print(f" Final model saved to {final_path}")

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Encoder(Training/Validation)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig_path = os.path.join(save_dir, f"loss_curve_{args.model_name}.png")
    plt.savefig(fig_path, dpi=160)
    print(f" Loss curve saved to {fig_path}")

    with open(os.path.join(save_dir, f"README_{args.model_name}.json"), "w") as f:
        json.dump({"paths": {"best": os.path.join(save_dir, f"best_model_{args.model_name}.pth"),
                             "final": final_path},
                   "note": "Checkpoint {'encoder','calib','config'}"}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", type=str, default="dataset")
    ap.add_argument("--K", type=int, default=90)
    ap.add_argument("--num_sequences", type=int, default=100000)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--padding_ratio", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--l2_normalize", action="store_true", default=False)

    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_squared_dist", action="store_true", default=False)
    ap.add_argument("--loss_mode", type=str, default="mse")     
    ap.add_argument("--lambda_triplet", type=float, default=0.2)
    ap.add_argument("--triplet_margin", type=float, default=0.1)
    ap.add_argument("--model_name", type=str, default="Encoder_Rebind")
    ap.add_argument("--save_dir", type=str, default="./trained_models")
    args = ap.parse_args()
    main(args)
