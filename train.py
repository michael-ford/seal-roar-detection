#!/usr/bin/env python3
"""train.py — SEARCH SPACE (agent modifies this file).

Baseline: lightweight CNN with focal loss on mel spectrograms.
The agent can modify anything here: model architecture, optimizer,
hyperparameters, augmentations, loss function, etc.

Run: uv run train.py
Output: prints metrics to stdout for results.tsv logging.
"""

import time

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from prepare import evaluate, load_splits

# ════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100          # Will be cut short by TIME_BUDGET
TIME_BUDGET = 180     # seconds (3 minutes)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75    # Weight for positive class (recall bias)

# ════════════════════════════════════════════════════════════════════════
# Augmentation
# ════════════════════════════════════════════════════════════════════════


class SpecAugment(nn.Module):
    """SpecAugment: time and frequency masking on spectrograms."""

    def __init__(self, freq_mask_param=15, time_mask_param=20, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        for _ in range(self.n_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.n_time_masks):
            x = self.time_mask(x)
        return x


def mixup(specs, labels, alpha=0.3):
    """Mixup augmentation: blend pairs of samples."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = specs.size(0)
    index = torch.randperm(batch_size)
    mixed_specs = lam * specs + (1 - lam) * specs[index]
    mixed_labels = lam * labels.float() + (1 - lam) * labels[index].float()
    return mixed_specs, mixed_labels


# ════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════


class SealRoarCNN(nn.Module):
    """Compact CNN with depthwise-separable convolutions.

    Faster to train = more epochs in time budget.
    Input: (batch, 1, 128, 126) mel spectrogram
    Output: (batch, 1) raw logits
    """

    def __init__(self):
        super().__init__()

        def dw_sep_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),  # depthwise
                nn.Conv2d(in_ch, out_ch, 1),  # pointwise
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


# ════════════════════════════════════════════════════════════════════════
# Loss
# ════════════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification.

    alpha: weight for positive class (higher = more recall)
    gamma: focusing parameter (higher = more focus on hard examples)
    """

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
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train():
    device = get_device()
    print(f"device={device}")

    # Load data
    train_ds, val_ds, _test_ds = load_splits()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = SealRoarCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params={n_params}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss()

    # Augmentation
    augment = SpecAugment()

    # Training loop
    start_time = time.time()
    best_val_f2 = 0.0
    best_metrics = None
    best_epoch = 0

    for epoch in range(EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET:
            print(f"time_budget_reached_at_epoch={epoch}")
            break

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for specs, labels in train_loader:
            specs = augment(specs)
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate every 2 epochs
        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            metrics = evaluate(model, val_ds, device=str(device))
            if metrics["f2"] > best_val_f2:
                best_val_f2 = metrics["f2"]
                best_metrics = metrics
                best_epoch = epoch
                torch.save(model.state_dict(), "results/best_model.pt")

    train_time = time.time() - start_time

    # Print results (parsed by experiment runner)
    if best_metrics is None:
        print("val_f2=0.0000")
        print("val_f1=0.0000")
        print("val_auc_roc=0.0000")
        print("val_precision=0.0000")
        print("val_recall=0.0000")
        print("val_threshold=0.5000")
    else:
        print(f"val_f2={best_metrics['f2']:.4f}")
        print(f"val_f1={best_metrics['f1']:.4f}")
        print(f"val_auc_roc={best_metrics['auc_roc']:.4f}")
        print(f"val_precision={best_metrics['precision']:.4f}")
        print(f"val_recall={best_metrics['recall']:.4f}")
        print(f"val_threshold={best_metrics['threshold']:.4f}")

    print(f"train_time={train_time:.1f}")
    print(f"best_epoch={best_epoch}")
    print(f"n_params={n_params}")


if __name__ == "__main__":
    train()
