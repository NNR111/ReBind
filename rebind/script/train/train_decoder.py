import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils import load_small_datasets, train_test_split

from rebind.models.rebind_model import BiGRUEncoder, MultiStageBidirectionalGRUDecoder

def main(args):
    datasets_dir = os.path.join(args.project_root, "datasets")
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    padded_len = int(args.seq_len * (1.0 + args.padding_ratio))
    datasets = load_small_datasets(datasets_dir, [args.K], padded_len)
    train_sets, test_sets = train_test_split(datasets)
    train_loader = DataLoader(train_sets[0], batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_sets[0],  batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.encoder_ckpt if os.path.isabs(args.encoder_ckpt) \
        else os.path.join(args.project_root, args.encoder_ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    enc_dim    = cfg.get("embedding_dim", args.embedding_dim)
    enc_hidden = cfg.get("enc_hidden_dim", args.enc_hidden_dim)
    enc_layers = cfg.get("enc_num_layers", args.enc_num_layers)
    enc_dropout = cfg.get("enc_dropout", args.enc_dropout)
    enc_l2 = cfg.get("enc_l2_norm", args.enc_l2_norm)

    encoder = BiGRUEncoder(
        embedding_dim=enc_dim,
        hidden_size=enc_hidden,
        num_layers=enc_layers,
        dropout=enc_dropout,
        l2_normalize=enc_l2
    ).to(device)
    state = ckpt["encoder"] if "encoder" in ckpt else ckpt
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder = MultiStageBidirectionalGRUDecoder(
        embedding_dim=enc_dim,
        hidden_dim=args.hidden_dim,
        output_dim=padded_len,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    best_hamming = float("inf")
    patience = 0
    train_losses = []

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total = 0.0
        for original, _, _ in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            original = original.to(device).float()
            with torch.no_grad():
                z = encoder(original)
            out = decoder(z)
            loss = 0.2 * mse(out, original) + 0.8 * bce(out, original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        avg = total / len(train_loader)
        train_losses.append(avg)
        print(f" Epoch {epoch}: Train Loss = {avg:.4f}")

        decoder.eval()
        tot_h, tot_n = 0.0, 0
        with torch.no_grad():
            for original, _, _ in test_loader:
                original = original.to(device).float()
                z = encoder(original)
                out = decoder(z)
                pred = (out > 0.5).float()
                tot_h += hamming_loss(pred.cpu(), original.cpu()) * original.size(0)
                tot_n += original.size(0)
        h = tot_h / tot_n
        print(f" Hamming Loss (test): {h:.4f}")
        scheduler.step(h)

        best_path = os.path.join(save_dir, args.best_save_name)
        if h < best_hamming:
            best_hamming = h
            torch.save(decoder.state_dict(), best_path)
            print(f" Best decoder saved (epoch {epoch}) → {best_path}")
            patience = 0
        else:
            patience += 1

        if patience >= args.early_stop_patience:
            print(" Early stopping (no improvement).")
            break

    final_path = os.path.join(save_dir, args.final_save_name)
    torch.save(decoder.state_dict(), final_path)
    print(f" Final decoder saved to {final_path}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Decoder Training Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig_path = os.path.join(save_dir, f"decoder_loss_curve_{args.hidden_dim}_k{args.K}.png")
    plt.savefig(fig_path)
    print(f" Loss curve saved to {fig_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", type=str, default="/home/coder/my code/invertible_separate")
    ap.add_argument("--K", type=int, default=90)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--padding_ratio", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--encoder_ckpt", type=str,
                    default="encoder_rebind.pth")

    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--enc_hidden_dim", type=int, default=256)
    ap.add_argument("--enc_num_layers", type=int, default=2)
    ap.add_argument("--enc_dropout", type=float, default=0.30)
    ap.add_argument("--enc_l2_norm", action="store_true", default=False)

    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.30)

    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--early_stop_patience", type=int, default=10)

    ap.add_argument("--best_save_name", type=str, default="best_decoder_rebind.pth")
    ap.add_argument("--final_save_name", type=str, default="final_decoder_rebind.pth")
    ap.add_argument("--save_dir", type=str, default="./trained_models")
    args = ap.parse_args()
    main(args)
