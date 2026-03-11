#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate ProtoNetMDAG with open-set rejection (Mahalanobis + quantile calibration).

Follows the window-smoothing + majority vote evaluation pattern from your original test code,
but removes all physics-related terms.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.manifold import TSNE

from protonet_mdag import ProtoNetMDAG


KNOWN_FILES: Dict[str, int] = {"0_normal.csv": 0, "2_pg.csv": 1, "3_pdb.csv": 2}
ALL_FILES_DEFAULT: Dict[str, int] = {"0_normal.csv": 0, "2_pg.csv": 1, "3_pdb.csv": 2, "1_ad.csv": -1, "4_pds.csv": -1}
LABEL_ORDER = [-1, 0, 1, 2]
CLASS_NAMES = ["Unknown", "Normal", "PG", "PDB"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_windows_from_files(
    data_path: str,
    file_dict: Dict[str, int],
    scaler,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for fname, label in file_dict.items():
        fp = os.path.join(data_path, fname)
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        data = scaler.transform(df.iloc[:, 1:].values)
        for i in range(0, len(data) - seq_len, stride):
            x_list.append(data[i:i+seq_len].T)
            y_list.append(label)
    X = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    if len(X) == 0:
        raise RuntimeError("No windows generated for evaluation. Check files/seq_len/stride.")
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


@torch.no_grad()
def calibrate_tau_md(model: ProtoNetMDAG, loader: DataLoader, quantile: float, device: torch.device) -> float:
    """Quantile calibration with a progress bar (known samples only)."""
    model.eval()
    scores = []
    for inputs, _labels in tqdm(loader, desc="Calibrating tau_md", leave=False):
        inputs = inputs.to(device)
        _z, _dists, _pred, s_md = model.forward_with_scores(inputs)
        scores.append(s_md.detach().cpu().numpy())
    s = np.concatenate(scores, axis=0)
    return float(np.quantile(s, quantile))


def window_vote_open_set(raw: List[Dict], win_size: int, tau_md: float) -> Tuple[np.ndarray, np.ndarray]:
    half = win_size // 2
    y_true, y_pred = [], []
    for i in range(len(raw)):
        w = raw[max(0, i-half):min(len(raw), i+half+1)]
        avg_md = sum(r["md"] for r in w) / len(w)
        if avg_md >= tau_md:
            final_cls = -1
        else:
            votes = [r["pred"] for r in w]
            final_cls = max(set(votes), key=votes.count)
        y_true.append(raw[i]["true"])
        y_pred.append(final_cls)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def plot_confusion_matrix_percent(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = (cm.astype(float) / row_sums) * 100.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_pct, interpolation="nearest")
    ax.set_title("Confusion Matrix (%)")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            ax.text(j, i, "0%" if val == 0 else f"{val:.2f}%", ha="center", va="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_tsne(z_feats: np.ndarray, y_true: np.ndarray, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    tsne = TSNE(n_components=2, perplexity=45, init="pca", random_state=42)
    emb = tsne.fit_transform(z_feats)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    label_to_idx = {-1: 0, 0: 1, 1: 2, 2: 3}
    colors = np.array([label_to_idx[int(v)] for v in y_true], dtype=int)

    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=18, alpha=0.8)
    ax.set_title("t-SNE of Embeddings (ProtoNetMDAG)")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def write_report_md(
    save_path: str,
    cfg: dict,
    tau_md: float,
    metrics: dict,
    cls_report: str,
    cm: np.ndarray,
    figure_paths: dict,
) -> None:
    ensure_dir(os.path.dirname(save_path))
    lines = []
    lines.append("# ProtoNetMDAG Evaluation Report\n")
    lines.append(f"**Timestamp:** {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Configuration\n")
    lines.append("```json\n" + json.dumps(cfg, indent=2) + "\n```\n")
    lines.append("## Open-set Calibration\n")
    lines.append(f"- MD threshold quantile: **{cfg['md_quantile']}**\n")
    lines.append(f"- Calibrated tau_md: **{tau_md:.6f}**\n")
    lines.append("\n## Metrics (macro)\n")
    lines.append(f"- Accuracy: **{metrics['accuracy']:.4f}**\n")
    lines.append(f"- Precision: **{metrics['precision']:.4f}**\n")
    lines.append(f"- Recall: **{metrics['recall']:.4f}**\n")
    lines.append(f"- F1-score: **{metrics['f1']:.4f}**\n")
    lines.append("\n## Classification Report\n")
    lines.append("```text\n" + cls_report + "\n```\n")
    lines.append("## Confusion Matrix (counts)\n")
    lines.append("```text\n" + np.array2string(cm) + "\n```\n")
    lines.append("## Figures\n")
    for k, p in figure_paths.items():
        lines.append(f"- {k}: `{p}`\n")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./dataset")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./result_statistics_mdag")
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--md_quantile", type=float, default=0.95)
    ap.add_argument("--win_size", type=int, default=11)
    ap.add_argument("--tsne", action="store_true")
    ap.add_argument("--unknown_files", type=str, default="1_ad.csv,4_pds.csv",
                    help="Comma-separated unknown CSV filenames (label=-1).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    scaler = ckpt.get("scaler", None)
    if scaler is None:
        raise RuntimeError("Checkpoint does not include scaler. Please retrain or provide scaler.")

    model = ProtoNetMDAG(input_dim=16, feature_dim=128, num_known_classes=3).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    run_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_dir = os.path.join(args.out_dir, run_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(out_dir)

    # Calibration loader (known)
    X_cal, y_cal = build_windows_from_files(args.data_dir, KNOWN_FILES, scaler, args.seq_len, args.stride)
    cal_loader = make_loader(X_cal, y_cal, args.batch_size)

    # Test loader (known + unknown)
    unknown_files = [s.strip() for s in args.unknown_files.split(",") if s.strip()]
    all_files = dict(KNOWN_FILES)
    for fn in unknown_files:
        all_files[fn] = -1

    X_test, y_test = build_windows_from_files(args.data_dir, all_files, scaler, args.seq_len, args.stride)
    test_loader = make_loader(X_test, y_test, args.batch_size)

    # Calibrate tau_md on known windows
    tau_md = calibrate_tau_md(model, cal_loader, quantile=args.md_quantile, device=device)

    # Evaluate
    raw = []
    z_all = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=True):
            inputs = inputs.to(device)
            z, _dists, pred, s_md = model.forward_with_scores(inputs)
            z_all.append(z.cpu().numpy())
            for i in range(len(labels)):
                raw.append({"true": int(labels[i].item()), "pred": int(pred[i].item()), "md": float(s_md[i].item())})

    y_true, y_pred = window_vote_open_set(raw, win_size=args.win_size, tau_md=tau_md)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {"accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cls_rep = classification_report(y_true, y_pred, labels=LABEL_ORDER, target_names=CLASS_NAMES, zero_division=0)

    fig_paths = {}
    cm_path = os.path.join(out_dir, "confusion_matrix_percent.png")
    plot_confusion_matrix_percent(cm, CLASS_NAMES, cm_path)
    fig_paths["confusion_matrix_percent"] = cm_path

    if args.tsne:
        z_feats = np.concatenate(z_all, axis=0)
        tsne_path = os.path.join(out_dir, "tsne.png")
        plot_tsne(z_feats, y_true=y_true, save_path=tsne_path)
        fig_paths["tsne"] = tsne_path

    report_cfg = {
        "data_dir": args.data_dir,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "md_quantile": args.md_quantile,
        "win_size": args.win_size,
        "labels": {"unknown": -1, "normal": 0, "pg": 1, "pdb": 2},
        "ckpt": args.ckpt,
        "unknown_files": unknown_files,
    }

    report_path = os.path.join(out_dir, "report.md")
    write_report_md(report_path, report_cfg, tau_md, metrics, cls_rep, cm, fig_paths)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"tau_md": tau_md, "metrics": metrics}, f, indent=2)

    print("\n=== ProtoNetMDAG Results ===")
    print(f"Saved report: {report_path}")
    print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    print(f"tau_md={tau_md:.6f}")


if __name__ == "__main__":
    main()
