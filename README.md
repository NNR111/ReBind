# REBIND: AN INVERTIBLE DEEP EMBEDDINGS FOR THE EDIT DISTANCE
1. learns an embedding where L2 distances predict Edit Distance (ED), and
2. can reconstruct the original binary sequence.

---

## Repository Structure

```
rebind/
├─ baselines/
│  ├─ cgk.py              # CGK ensemble baseline
│  ├─ cnned.py            # CNN-ED baseline
│  ├─ gru.py              # GRU baseline
│  └─ transformer.py      # Transformer baseline
├─ models/
│  └─ rebind.py           # ReBind model 
├─ script/
│  ├─ train/              # training entrypoints 
│  └─ eval/               # evaluation entrypoints 
├─ utils.py               # Dataset loaders 
└─ readme.md
```

## Evaluation

Fidelity (RMSE/MAE/Pearson/Spearman), Ranking Consistency (Triplet-Acc), Invertibility (Hamming), and Timing (encode-only):

```bash
python script/eval/eval_rebind.py \
  --project_root . \
  --dataset_folder ./datasets \
  --ckpt ./checkpoints/best_rebind_k90.pth \
  --k 90 --seq_len 100 --padding_ratio 0.3 \
  --eval_batch_size 256 \
  --out_dir ./results/rebind_k90
```


