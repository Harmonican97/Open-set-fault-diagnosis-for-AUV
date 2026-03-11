#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ProtoNetMDAG (physics removed) on known classes.

Keeps the core logic from the original training script: WeightedRandomSampler,
OneCycleLR, v15 loss + topology loss, and tqdm progress bars.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from protonet_mdag import ProtoNetMDAG, compute_topology_loss, get_haizhe_similarity_matrix


KNOWN_FILE_MAP_DEFAULT: Dict[str, int] = {
    "0_normal.csv": 0,
    "2_pg.csv": 1,
    "3_pdb.csv": 2,
}


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    seq_len: int
    stride: int
    batch_size: int
    epochs: int
    lr: float
    max_lr: float
    weight_decay: float
    margin: float
    lambda_v15: float
    lambda_topo: float
    val_frac: float
    seed: int
    weighted_sampler: bool


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_known_csv_series(data_path: str, file_map: Dict[str, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data_list, label_list = [], []
    for file_name, label in file_map.items():
        fp = os.path.join(data_path, file_name)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"File not found: {fp}")
        df = pd.read_csv(fp)
        feats = df.iloc[:, 1:].values  # drop time col
        data_list.append(feats)
        label_list.append(np.full(len(feats), label, dtype=np.int64))
    return data_list, label_list


def make_windows(
    data_list: List[np.ndarray],
    label_list: List[np.ndarray],
    scaler: StandardScaler,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_frames, y_frames = [], []
    for data, labels in zip(data_list, label_list):
        std_data = scaler.transform(data)
        for i in range(0, len(std_data) - seq_len, stride):
            x_frames.append(std_data[i:i+seq_len].T)  # (C,L)
            y_frames.append(int(labels[i]))
    X = np.asarray(x_frames, dtype=np.float32)
    y = np.asarray(y_frames, dtype=np.int64)
    if len(X) == 0:
        raise RuntimeError("No training windows generated. Check seq_len/stride and CSV length.")
    return X, y


def split_train_val(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_frac <= 0.0:
        return X, y, X[:0], y[:0]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    (train_idx, val_idx), = sss.split(X, y)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, weighted: bool, seed: int, shuffle: bool = True) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    if not weighted:
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle, drop_last=False)

    class_counts = np.bincount(y)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    weights = class_weights[y]
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(weights), generator=g)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, sampler=sampler, drop_last=False)


def train_one_epoch(
    model: ProtoNetMDAG,
    loader: DataLoader,
    optimizer,
    scheduler,
    sim_mat: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch+1}/{cfg.epochs}", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        z, dists = model(inputs)

        l_cls = F.cross_entropy(-dists, labels)
        l_v15 = model.compute_v15_loss(z, dists, labels, margin=cfg.margin)
        l_topo = compute_topology_loss(model.prototypes, sim_mat)

        loss = l_cls + cfg.lambda_v15 * l_v15 + cfg.lambda_topo * l_topo
        loss.backward()
        optimizer.step()
        scheduler.step()

        running += float(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})

    return running / max(1, len(loader))


@torch.no_grad()
def update_running_var_with_pbar(model: ProtoNetMDAG, loader: DataLoader, device: torch.device, eps: float = 1e-5) -> None:
    """Re-estimate running_var with a progress bar (for Mahalanobis scoring)."""
    model.eval()
    all_z, all_l = [], []
    for inputs, labels in tqdm(loader, desc="Update running_var", leave=False):
        inputs = inputs.to(device)
        z, _ = model(inputs)
        all_z.append(z.detach().cpu())
        all_l.append(labels.detach().cpu())

    all_z = torch.cat(all_z, dim=0)
    all_l = torch.cat(all_l, dim=0)
    num_classes = model.prototypes.size(0)
    for c in range(num_classes):
        z_c = all_z[all_l == c]
        if z_c.size(0) > 1:
            model.running_var[c] = torch.var(z_c, dim=0) + eps


@torch.no_grad()
def eval_one_epoch(
    model: ProtoNetMDAG,
    loader: DataLoader,
    sim_mat: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    running = 0.0
    pbar = tqdm(loader, desc=f"Val   {epoch+1}/{cfg.epochs}", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        z, dists = model(inputs)
        l_cls = F.cross_entropy(-dists, labels)
        l_v15 = model.compute_v15_loss(z, dists, labels, margin=cfg.margin)
        l_topo = compute_topology_loss(model.prototypes, sim_mat)
        loss = l_cls + cfg.lambda_v15 * l_v15 + cfg.lambda_topo * l_topo

        running += float(loss.item())
        pbar.set_postfix({"val_loss": f"{loss.item():.3f}"})

    return running / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./dataset")
    ap.add_argument("--out_dir", type=str, default="./runs_mdag")
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--margin", type=float, default=20.0)
    ap.add_argument("--lambda_v15", type=float, default=1.2)
    ap.add_argument("--lambda_topo", type=float, default=0.3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_weighted_sampler", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        lambda_v15=args.lambda_v15,
        lambda_topo=args.lambda_topo,
        val_frac=args.val_frac,
        seed=args.seed,
        weighted_sampler=(not args.no_weighted_sampler),
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"ProtoNetMDAG_seq{cfg.seq_len}_str{cfg.stride}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Data -> scaler -> windows
    raw_data, raw_labels = load_known_csv_series(cfg.data_dir, KNOWN_FILE_MAP_DEFAULT)
    scaler = StandardScaler()
    scaler.fit(np.vstack(raw_data))
    X, y = make_windows(raw_data, raw_labels, scaler, cfg.seq_len, cfg.stride)

    Xtr, ytr, Xva, yva = split_train_val(X, y, cfg.val_frac, cfg.seed)
    train_loader = make_loader(Xtr, ytr, cfg.batch_size, cfg.weighted_sampler, cfg.seed, shuffle=True)
    val_loader = make_loader(Xva, yva, cfg.batch_size, weighted=False, seed=cfg.seed, shuffle=False) if len(Xva) else None

    # Model
    model = ProtoNetMDAG(input_dim=X.shape[1], feature_dim=128, num_known_classes=3).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.max_lr, steps_per_epoch=len(train_loader), epochs=cfg.epochs)

    sim_mat = get_haizhe_similarity_matrix(3).to(device)

    history = []
    best_val = float("inf")
    best_state = None

    epoch_pbar = tqdm(range(cfg.epochs), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, sim_mat, cfg, device, epoch)
        va_loss = eval_one_epoch(model, val_loader, sim_mat, cfg, device, epoch) if val_loader is not None else tr_loss
        history.append({"epoch": epoch+1, "train_loss": tr_loss, "val_loss": va_loss})
        epoch_pbar.set_postfix({"train_loss": f"{tr_loss:.3f}", "val_loss": f"{va_loss:.3f}"})

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # Re-estimate running_var for Mahalanobis scoring on all known windows
    full_loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    update_running_var_with_pbar(model, full_loader, device=device)

    ckpt_path = os.path.join(run_dir, "best_model.pth")
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "config": asdict(cfg),
            "run_dir": run_dir,
        },
        ckpt_path,
    )

    with open(os.path.join(run_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nSaved best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
