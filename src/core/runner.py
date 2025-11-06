import json
from pathlib import Path
import torch
import torch.nn as nn

from src.core.utils import set_seed, get_device, ensure_dir
from src.data.hyperspectral import load_hsi_as_graph
from src.models.deepgcn import TinyDeepGCN
from src.explainers.integrated_gradients import explain_ig
from src.explainers.saliency import explain_saliency
from src.explainers.gradcam import explain_gradcam_hidden

def train(model, x, adj, y, device, epochs=60, lr=5e-3, wd=5e-4, patience=10, out_dir="results"):
    model.to(device)
    x, adj, y = x.to(device), adj.to(device), y.to(device).long()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_ep, wait = -1.0, -1, 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, adj)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

        improved = acc > best_acc
        if improved:
            best_acc, best_ep, wait = acc, ep, 0
            torch.save({"state": model.state_dict(), "acc": best_acc, "epoch": ep},
                       Path(out_dir) / "checkpoint.pt")
        else:
            wait += 1

        print(f"[{ep:03d}] loss={loss.item():.4f} acc={acc:.4f}{' *' if improved else ''}")
        if wait >= patience:
            print(f"Early stop at {ep} (best {best_acc:.4f} @ {best_ep})")
            break

    with open(Path(out_dir) / "train_summary.json", "w") as f:
        json.dump({"best_acc": best_acc, "best_epoch": best_ep}, f, indent=2)
    return best_acc

def summarize_band_importance(attributions, y, band_names, topk, out_dir):
    import numpy as np, pandas as pd
    atts = attributions.detach().cpu()  # [N, B]
    y_cpu = y.detach().cpu().numpy()
    cols = band_names if band_names is not None else [f"band_{i}" for i in range(atts.shape[1])]
    records = []
    n_classes = int(y.max()) + 1
    for c in range(n_classes):
        mask = (y_cpu == c)
        if mask.sum() == 0:
            continue
        mean_band = atts[mask].mean(dim=0).numpy()
        top_idx = np.argsort(-mean_band)[:topk]
        records.append({
            "class": int(c),
            "topk_bands": [cols[i] for i in top_idx],
            "topk_scores": [float(mean_band[i]) for i in top_idx]
        })
    pd.DataFrame(records).to_csv(Path(out_dir) / "band_ranking.csv", index=False)
    print(f"[OK] Saved top-{topk} band ranking → {Path(out_dir) / 'band_ranking.csv'}")

def run(args):
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out_dir)

    # Load graph (x:[N,B], adj:[N,N], y:[N])
    x, adj, y, band_names = load_hsi_as_graph(
        dataset=args.dataset, data_dir=args.data_dir,
        mat_x=args.mat_x or None, mat_y=args.mat_y or None,
        mat_x_key=args.mat_x_key, mat_y_key=args.mat_y_key,
        knn=args.knn, device=device
    )
    n_nodes, feat_dim = x.shape
    n_classes = int(y.max().item()) + 1

    # Model
    model = TinyDeepGCN(
        in_dim=feat_dim, hid=args.hid, out_dim=n_classes,
        n_layers=args.layers, dropout=args.dropout,
        norm_mode=args.norm_mode, norm_scale=args.norm_scale
    ).to(device)

    if args.mode == "train":
        train(model, x, adj, y, device, epochs=args.epochs, lr=args.lr, wd=args.wd,
              patience=args.patience, out_dir=args.out_dir)

    elif args.mode == "explain":
        # Load checkpoint if present
        ckpt = Path(args.out_dir) / "checkpoint.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device)["state"])
            print(f"[OK] Loaded checkpoint: {ckpt}")
        model.eval()

        if args.explainer == "ig":
            attributions = explain_ig(model, x, adj, y, device=device)             # [N,B]
        elif args.explainer == "saliency":
            attributions = explain_saliency(model, x, adj, y, device=device)       # [N,B]
        elif args.explainer == "gradcam":
            attributions = explain_gradcam_hidden(model, x, adj, y, device=device) # [N,B]-approx
        else:
            raise ValueError("Unsupported explainer")

        summarize_band_importance(attributions, y, band_names, args.topk, args.out_dir)
    else:
        raise ValueError("Unknown mode")
