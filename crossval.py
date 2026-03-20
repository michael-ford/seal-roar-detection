#!/usr/bin/env python3
"""Cross-validation across time periods.

Rotates pre-computed spectrogram blocks through train/val/test roles.
No reprocessing needed — uses existing .pt files.

Usage: uv run crossval.py
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, TensorDataset

from prepare import DATA_SPLITS, evaluate

# ════════════════════════════════════════════════════════════════════════
# Folds: rotate 3 time-period blocks
# ════════════════════════════════════════════════════════════════════════

FOLDS = [
    {"train": "train", "val": "val", "test": "test", "name": "Train=Sep1, Val=Sep2, Test=Oct"},
    {"train": "val", "val": "test", "test": "train", "name": "Train=Sep2, Val=Oct, Test=Sep1"},
    {"train": "test", "val": "train", "test": "val", "name": "Train=Oct, Val=Sep1, Test=Sep2"},
]

# ════════════════════════════════════════════════════════════════════════
# Model (same as deployed depthwise-sep CNN)
# ════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
TIME_BUDGET = 180
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75


class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(15)
        self.time_mask = torchaudio.transforms.TimeMasking(20)

    def forward(self, x):
        for _ in range(2):
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x


class SealRoarCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def dw_sep_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            dw_sep_block(32, 64),
            dw_sep_block(64, 128),
            dw_sep_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# ════════════════════════════════════════════════════════════════════════
# Training + evaluation for one fold
# ════════════════════════════════════════════════════════════════════════


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_block(split_name):
    """Load a pre-computed .pt file as a TensorDataset."""
    data = torch.load(DATA_SPLITS / f"{split_name}.pt", weights_only=True)
    return data["spectrograms"].float(), data["labels"]


def run_fold(fold_cfg, device):
    """Train and evaluate one fold. Returns metrics dict."""
    train_specs, train_labels = load_block(fold_cfg["train"])
    val_specs, val_labels = load_block(fold_cfg["val"])
    test_specs, test_labels = load_block(fold_cfg["test"])

    train_ds = TensorDataset(train_specs, train_labels)
    val_ds = TensorDataset(val_specs, val_labels)
    test_ds = TensorDataset(test_specs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = SealRoarCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss()
    augment = SpecAugment()

    start = time.time()
    best_val_f2 = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        if time.time() - start > TIME_BUDGET:
            break

        model.train()
        for specs, labels in train_loader:
            specs = augment(specs)
            specs, labels = specs.to(device), labels.to(device)
            loss = criterion(model(specs).squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_metrics = evaluate(model, val_ds, device=str(device))
        if val_metrics["f2"] > best_val_f2:
            best_val_f2 = val_metrics["f2"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate best model on TEST set
    if best_state:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_ds, device=str(device))
    val_metrics = evaluate(model, val_ds, device=str(device))

    return {
        "val": val_metrics,
        "test": test_metrics,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_time": time.time() - start,
    }


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════


def main():
    device = get_device()
    print(f"Device: {device}\n")

    all_results = []

    for i, fold in enumerate(FOLDS):
        print(f"{'='*60}")
        print(f"FOLD {i+1}: {fold['name']}")
        print(f"{'='*60}")
        result = run_fold(fold, device)
        all_results.append(result)

        print(f"  Train: {result['train_size']} samples")
        print(f"  Val:   {result['val_size']} samples")
        print(f"  Test:  {result['test_size']} samples")
        print(f"  Time:  {result['train_time']:.0f}s")
        print(f"  --- Val metrics ---")
        for k, v in result["val"].items():
            print(f"    {k}: {v:.4f}")
        print(f"  --- Test metrics ---")
        for k, v in result["test"].items():
            print(f"    {k}: {v:.4f}")
        print()

    # Aggregate
    print(f"{'='*60}")
    print("AGGREGATE (mean ± std across 3 folds)")
    print(f"{'='*60}")
    for split_name in ["val", "test"]:
        print(f"\n  {split_name.upper()} metrics:")
        for metric in ["f2", "f1", "auc_roc", "precision", "recall"]:
            values = [r[split_name][metric] for r in all_results]
            print(f"    {metric:12s}: {np.mean(values):.4f} ± {np.std(values):.4f}  [{', '.join(f'{v:.3f}' for v in values)}]")


if __name__ == "__main__":
    main()
