import os, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_triplet_datasets, train_test_split
from rebind.baselines.cnned import (
    CNNEDEncoder, DistanceCalib, compute_dist, count_params, get_last_conv_channels
)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    dataset_folder = args.dataset_folder or os.path.join(args.project_root, "datasets")
    save_dir = ensure_dir(args.save_dir)

    datasets = load_triplet_datasets(dataset_folder, [args.k], args.num_sequences, padded_len)
    train_sets, test_sets = train_test_split(datasets)
    train_loader = DataLoader(train_sets[0], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(test_sets[0],  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    kernels  = tuple(int(x) for x in args.kernels.split(",") if x.strip())
    pools    = tuple(int(x) for x in args.pools.split(",") if x.strip())

    encoder = CNNEDEncoder(
        embedding_dim=args.embedding_dim,
        channels=channels, kernels=kernels, pools=pools,
        dropout=args.dropout, l2_normalize=args.l2_normalize
    ).to(device)
    calib = DistanceCalib().to(device)

    print(f"[CNN-ED] params = {count_params(encoder)/1e6:.3f} M")


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
            a = a.to(device).float(); p = p.to(device).float(); n = n.to(device).float()
            d_ap = d_ap.to(device).float().squeeze()
            d_an = d_an.to(device).float().squeeze()
            d_pn = d_pn.to(device).float().squeeze()

            z_a = encoder(a); z_p = encoder(p); z_n = encoder(n)
            r_ap = compute_dist(z_a, z_p, squared=args.use_squared_dist)
            r_an = compute_dist(z_a, z_n, squared=args.use_squared_dist)
            r_pn = compute_dist(z_p, z_n, squared=args.use_squared_dist)

            p_ap = calib(r_ap); p_an = calib(r_an); p_pn = calib(r_pn)
            loss_reg = (mse(p_ap, d_ap) + mse(p_an, d_an) + mse(p_pn, d_pn)) / 3.0
            trip1 = F.relu(p_ap - p_an + args.triplet_margin)
            trip2 = F.relu(p_pn - p_an + args.triplet_margin)
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
        print(f"✅ epoch {epoch} | train loss {avg_train:.4f}")

        encoder.eval(); calib.eval()
        with torch.no_grad():
            total_val = 0.0
            for a, p, n, d_ap, d_an, d_pn in val_loader:
                a = a.to(device).float(); p = p.to(device).float(); n = n.to(device).float()
                d_ap = d_ap.to(device).float().squeeze()
                d_an = d_an.to(device).float().squeeze()
                d_pn = d_pn.to(device).float().squeeze()

                z_a = encoder(a); z_p = encoder(p); z_n = encoder(n)
                r_ap = compute_dist(z_a, z_p, squared=args.use_squared_dist)
                r_an = compute_dist(z_a, z_n, squared=args.use_squared_dist)
                r_pn = compute_dist(z_p, z_n, squared=args.use_squared_dist)

                p_ap = calib(r_ap); p_an = calib(r_an); p_pn = calib(r_pn)
                val_loss = (nn.functional.mse_loss(p_ap, d_ap) +
                            nn.functional.mse_loss(p_an, d_an) +
                            nn.functional.mse_loss(p_pn, d_pn)) / 3.0
                total_val += float(val_loss.item())

            avg_val = total_val / max(1, len(val_loader))
            val_losses.append(avg_val)
            print(f"🔵 val loss {avg_val:.4f}")
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                ckpt_path = os.path.join(save_dir, f"best_model_{args.model_name}.pth")
                torch.save({
                    "encoder": encoder.state_dict(),
                    "calib": calib.state_dict(),
                    "config": {
                        "embedding_dim": args.embedding_dim,
                        "channels": ",".join(map(str, channels)),
                        "kernels": ",".join(map(str, kernels)),
                        "pools": ",".join(map(str, pools)),
                        "use_squared_dist": args.use_squared_dist,
                        "lambda_triplet": args.lambda_triplet,
                        "triplet_margin": args.triplet_margin,
                        "last_conv_channels": get_last_conv_channels(encoder),
                    }
                }, ckpt_path)
                print(f"💾 saved best → {ckpt_path}")

    final_path = os.path.join(save_dir, f"final_model_{args.model_name}.pth")
    torch.save({
        "encoder": encoder.state_dict(),
        "calib": calib.state_dict(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "channels": ",".join(map(str, channels)),
            "kernels": ",".join(map(str, kernels)),
            "pools": ",".join(map(str, pools)),
            "last_conv_channels": get_last_conv_channels(encoder),
        }
    }, final_path)
    print(f"📦 saved final → {final_path}")

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE calib)")
    plt.title("CNN-ED + Calibrator")
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig_path = os.path.join(save_dir, f"loss_curve_{args.model_name}.png")
    plt.savefig(fig_path, dpi=160)
    print(f"📈 curve → {fig_path}")

    with open(os.path.join(save_dir, f"README_{args.model_name}.json"), "w") as f:
        json.dump({
            "paths": {"best": os.path.join(save_dir, f"best_model_{args.model_name}.pth"),
                      "final": final_path},
            "note": "Checkpoint {'encoder','calib','config'}"
        }, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train") 
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--dataset_folder", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)


    ap.add_argument("--k", type=int, default=90)
    ap.add_argument("--num_sequences", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--padding_ratio", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)


    ap.add_argument("--channels", type=str, default="64,128,256")
    ap.add_argument("--kernels",  type=str, default="5,5,3")
    ap.add_argument("--pools",    type=str, default="2,2,2")
    ap.add_argument("--dropout",  type=float, default=0.30)
    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--l2_normalize", action="store_true", default=False)


    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_triplet", type=float, default=0.2)
    ap.add_argument("--triplet_margin", type=float, default=0.1)
    ap.add_argument("--use_squared_dist", action="store_true", default=False)


    ap.add_argument("--save_dir", type=str, default="./trained_models")
    ap.add_argument("--model_name", type=str, default="CNNED_K90")
    args = ap.parse_args()
    main(args)


