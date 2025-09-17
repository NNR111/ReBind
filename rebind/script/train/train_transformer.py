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
from rebind.baselines.transformer import (
    TransformerEDEncoder, DistanceCalib, cosine_distance,
    vicreg_regularizer, count_params
)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Tpad = int(args.seq_len * (1.0 + args.padding_ratio))
    dataset_folder = args.dataset_folder or os.path.join(args.project_root, "datasets")
    save_dir = ensure_dir(args.save_dir)


    datasets = load_triplet_datasets(dataset_folder, [args.k], args.num_sequences, Tpad)
    train_sets, val_sets = train_test_split(datasets)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_sets[0], batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_sets[0],   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)


    enc = TransformerEDEncoder(
        embedding_dim=args.embedding_dim, model_dim=args.model_dim,
        nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout, l2_normalize=True
    ).to(device)
    cal = DistanceCalib().to(device)
    print(f"[Transformer-ED] params = {count_params(enc)/1e6:.3f} M")


    params = list(enc.parameters()) + list(cal.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)
    mse = nn.MSELoss()

    best_val = float("inf")
    tr_losses, va_losses = [], []

    for ep in range(1, args.epochs + 1):
        enc.train(); cal.train()
        tot = 0.0
        for a, p, n, d_ap, d_an, d_pn in tqdm(train_loader, desc=f"[Ep {ep}]"):
            a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
            d_ap=d_ap.to(device).squeeze().float()
            d_an=d_an.to(device).squeeze().float()
            d_pn=d_pn.to(device).squeeze().float()

            za, zp, zn = enc(a), enc(p), enc(n)
            rap = cosine_distance(za, zp)
            ran = cosine_distance(za, zn)
            rpn = cosine_distance(zp, zn)

            pap, pan, ppn = cal(rap), cal(ran), cal(rpn)

            loss_reg = (mse(pap, d_ap) + mse(pan, d_an) + mse(ppn, d_pn)) / 3.0
            trip = F.relu(pap - pan + args.triplet_margin) + F.relu(ppn - pan + args.triplet_margin)
            vloss, closs = vicreg_regularizer(torch.cat([za, zp, zn], dim=0))
            loss = loss_reg + args.lambda_triplet * trip.mean() + args.lambda_var * vloss + args.lambda_cov * closs

            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step(); tot += float(loss.item())

        tr_loss = tot / max(1, len(train_loader))
        tr_losses.append(tr_loss)
        print(f" train loss: {tr_loss:.4f}")

        # ---- Validation ----
        enc.eval(); cal.eval()
        with torch.no_grad():
            vtot = 0.0
            for a, p, n, d_ap, d_an, d_pn in val_loader:
                a=a.to(device).float(); p=p.to(device).float(); n=n.to(device).float()
                d_ap=d_ap.to(device).squeeze().float()
                d_an=d_an.to(device).squeeze().float()
                d_pn=d_pn.to(device).squeeze().float()

                za, zp, zn = enc(a), enc(p), enc(n)
                rap = cosine_distance(za, zp); ran = cosine_distance(za, zn); rpn = cosine_distance(zp, zn)
                pap, pan, ppn = cal(rap), cal(ran), cal(rpn)
                vloss = (mse(pap, d_ap) + mse(pan, d_an) + mse(ppn, d_pn)) / 3.0
                vtot += float(vloss.item())

            va_loss = vtot / max(1, len(val_loader))
            va_losses.append(va_loss)
            print(f" val loss: {va_loss:.4f}")
            sch.step()

            if va_loss < best_val:
                best_val = va_loss
                ckpt_path = os.path.join(save_dir, f"best_model_{args.model_name}.pth")
                torch.save({
                    "encoder": enc.state_dict(),
                    "calib": cal.state_dict(),
                    "config": {
                        "embedding_dim": args.embedding_dim,
                        "model_dim": args.model_dim, "nhead": args.nhead, "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "lambda_triplet": args.lambda_triplet, "triplet_margin": args.triplet_margin,
                        "lambda_var": args.lambda_var, "lambda_cov": args.lambda_cov
                    }
                }, ckpt_path)
                print(f" saved best → {ckpt_path}")

    # Save final + curve
    final_path = os.path.join(save_dir, f"final_model_{args.model_name}.pth")
    torch.save({"encoder": enc.state_dict(), "calib": cal.state_dict()}, final_path)
    print(f" saved final → {final_path}")

    plt.figure(figsize=(6,4))
    import matplotlib.pyplot as plt
    plt.plot(tr_losses, label="Train"); plt.plot(va_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss (calib MSE)"); plt.title("Transformer-ED + Calibrator")
    plt.legend(); plt.grid(True); plt.tight_layout()
    figp = os.path.join(save_dir, f"loss_curve_{args.model_name}.png")
    plt.savefig(figp, dpi=160); print(f" curve → {figp}")

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
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)


    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_triplet", type=float, default=0.4)
    ap.add_argument("--triplet_margin", type=float, default=0.3)
    ap.add_argument("--lambda_var", type=float, default=0.01)
    ap.add_argument("--lambda_cov", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)


    ap.add_argument("--save_dir", type=str, default="./trained_models")
    ap.add_argument("--model_name", type=str, default="TFED_K90")
    args = ap.parse_args()
    main(args)