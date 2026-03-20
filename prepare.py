#!/usr/bin/env python3
"""prepare.py — LOCKED data pipeline and evaluation function.

Agent must NOT modify this file.

Run:
    uv run prepare.py           — process annotations + WAVs → spectrograms + splits
    uv run prepare.py --stats   — show dataset statistics

Import:
    from prepare import SealRoarDataset, load_splits, evaluate
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

# ════════════════════════════════════════════════════════════════════════
# Configuration (immutable)
# ════════════════════════════════════════════════════════════════════════

ORIG_SAMPLE_RATE = 36000
TARGET_SAMPLE_RATE = 16000
CLIP_DURATION = 4.0  # seconds
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 50   # Hz — harbour seal roar lower bound
FMAX = 4000 # Hz — upper bound
NEG_POS_RATIO = 3
NEG_BUFFER_SEC = 6.0  # seconds buffer around annotations for negative sampling
RANDOM_SEED = 42

# Temporal split boundaries (UTC)
TRAIN_END = datetime(2017, 9, 21)   # Train: Sep 1–20
VAL_END = datetime(2017, 10, 1)     # Val: Sep 21–30
                                     # Test: Oct 1–15

# Paths
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
ANNOTATIONS_FILE = ROOT / "annotations" / "Lasqueti_Nov2017Pv_annotationsMar19_RSL.csv"

# ════════════════════════════════════════════════════════════════════════
# Audio processing
# ════════════════════════════════════════════════════════════════════════

# Lazy-initialized transforms (created once, reused)
_resampler = None
_mel_transform = None


def _get_resampler():
    global _resampler
    if _resampler is None:
        _resampler = torchaudio.transforms.Resample(ORIG_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    return _resampler


def _get_mel_transform():
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=2.0,  # Power spectrogram (needed for PCEN)
        )
    return _mel_transform


def apply_pcen(mel_spec, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    """Per-Channel Energy Normalization (Wang et al., 2017).

    Args:
        mel_spec: (channels, n_mels, time) power mel spectrogram
    Returns:
        PCEN-normalized spectrogram, same shape
    """
    M = torch.zeros_like(mel_spec)
    M[..., 0] = mel_spec[..., 0]
    for t in range(1, mel_spec.shape[-1]):
        M[..., t] = (1 - s) * M[..., t - 1] + s * mel_spec[..., t]
    smooth = (M + eps).pow(alpha)
    return (mel_spec / smooth + delta).pow(r) - delta**r


def parse_wav_timestamp(filepath):
    """Extract UTC start time from WAV filename.

    Format: DFOCRP.AM107-LQ.ST604553225.YYYYMMDD_HHMMSSZ.wav
    """
    stem = Path(filepath).stem
    ts_part = stem.split(".")[3].rstrip("Z")
    return datetime.strptime(ts_part, "%Y%m%d_%H%M%S")


def build_wav_index(raw_dir):
    """Scan WAV files and build a sorted index of (start_time, duration_sec, filepath).

    Returns list sorted by start_time.
    """
    wav_files = sorted(raw_dir.glob("*.wav"))
    index = []
    for wf in wav_files:
        start = parse_wav_timestamp(wf)
        info = torchaudio.info(wf)
        duration = info.num_frames / info.sample_rate
        index.append((start, duration, wf))
    index.sort(key=lambda x: x[0])
    return index


def find_wav_for_time(wav_index, utc_time):
    """Find the WAV file containing a given UTC time.

    Returns (wav_start, wav_duration, wav_path, offset_sec) or None.
    """
    for start, duration, path in reversed(wav_index):
        if start <= utc_time:
            offset = (utc_time - start).total_seconds()
            if offset <= duration:
                return start, duration, path, offset
            return None  # Falls in a gap between recordings
    return None


def extract_clip(wav_path, offset_sec, wav_duration):
    """Extract a 4-second audio clip, resample to 16 kHz.

    The clip is centered on offset_sec, clamped to file bounds.
    Returns waveform tensor (1, n_samples) or None on error.
    """
    # Center the clip on the offset
    clip_start = offset_sec - CLIP_DURATION / 2.0
    # Clamp to file bounds
    clip_start = max(0.0, min(clip_start, wav_duration - CLIP_DURATION))

    frame_offset = int(clip_start * ORIG_SAMPLE_RATE)
    num_frames = int(CLIP_DURATION * ORIG_SAMPLE_RATE)

    try:
        waveform, sr = torchaudio.load(wav_path, frame_offset=frame_offset, num_frames=num_frames)
    except Exception as e:
        print(f"  WARNING: Failed to load {wav_path} at offset {clip_start:.1f}s: {e}")
        return None

    if waveform.shape[1] < num_frames * 0.9:  # Less than 90% of expected length
        return None

    # Pad if slightly short
    if waveform.shape[1] < num_frames:
        pad = num_frames - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    # Resample
    waveform = _get_resampler()(waveform)
    return waveform


def compute_spectrogram(waveform):
    """Compute mel spectrogram + PCEN normalization.

    Args:
        waveform: (1, n_samples) tensor
    Returns:
        spectrogram: (1, N_MELS, T) tensor
    """
    mel_spec = _get_mel_transform()(waveform)  # (1, N_MELS, T)
    spec = apply_pcen(mel_spec)
    return spec


# ════════════════════════════════════════════════════════════════════════
# Annotation processing
# ════════════════════════════════════════════════════════════════════════


def load_annotations(csv_path):
    """Load annotations, keeping only confirmed 'Pv roar'.

    Returns list of dicts with keys: id, utc, duration, f1, f2, label
    """
    annotations = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["Label"].strip()
            if label != "Pv roar":
                continue
            utc = datetime.strptime(row["UTC"].strip(), "%Y-%m-%d %H:%M:%S.%f")
            annotations.append(
                {
                    "id": int(row["Id"]),
                    "utc": utc,
                    "duration": float(row["Duration"]),
                    "label": 1,
                }
            )
    return annotations


def get_split(utc):
    """Assign temporal split based on UTC date."""
    if utc < TRAIN_END:
        return "train"
    elif utc < VAL_END:
        return "val"
    else:
        return "test"


# ════════════════════════════════════════════════════════════════════════
# Negative sampling
# ════════════════════════════════════════════════════════════════════════


def get_annotations_in_file(wav_start, wav_duration, annotations):
    """Get annotation intervals (as offsets within a WAV file) that overlap this file."""
    wav_end_utc = wav_start + __import__("datetime").timedelta(seconds=wav_duration)
    intervals = []
    for ann in annotations:
        ann_start = (ann["utc"] - wav_start).total_seconds()
        ann_end = ann_start + ann["duration"]
        # Check overlap with file
        if ann_end > 0 and ann_start < wav_duration:
            intervals.append((max(0, ann_start), min(wav_duration, ann_end)))
    return intervals


def find_free_intervals(wav_duration, occupied_intervals, buffer=NEG_BUFFER_SEC):
    """Find time intervals not covered by occupied intervals (with buffer).

    Returns list of (start_sec, end_sec) that are free for negative sampling.
    """
    # Expand intervals by buffer
    blocked = []
    for start, end in occupied_intervals:
        blocked.append((max(0, start - buffer), min(wav_duration, end + buffer)))

    # Merge overlapping blocked intervals
    blocked.sort()
    merged = []
    for start, end in blocked:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Find free intervals (must fit a full clip)
    free = []
    prev_end = 0.0
    for start, end in merged:
        if start - prev_end >= CLIP_DURATION:
            free.append((prev_end, start))
        prev_end = end
    if wav_duration - prev_end >= CLIP_DURATION:
        free.append((prev_end, wav_duration))

    return free


def sample_negatives(wav_index, annotations, rng):
    """Sample negative clips from unannotated periods.

    Samples NEG_POS_RATIO negatives per positive, distributed across splits.
    Returns list of dicts with keys: utc, wav_path, offset_sec, split, label
    """
    # Count positives per split
    split_counts = {"train": 0, "val": 0, "test": 0}
    for ann in annotations:
        split_counts[get_split(ann["utc"])] += 1

    # Target negatives per split
    targets = {s: c * NEG_POS_RATIO for s, c in split_counts.items()}

    # Build pool of free intervals per split
    split_pools = {"train": [], "val": [], "test": []}
    for wav_start, wav_dur, wav_path in wav_index:
        split = get_split(wav_start)
        occupied = get_annotations_in_file(wav_start, wav_dur, annotations)
        free = find_free_intervals(wav_dur, occupied)
        for interval_start, interval_end in free:
            # Weight by interval length (more time = more samples)
            split_pools[split].append((wav_start, wav_path, interval_start, interval_end))

    # Sample from pools
    negatives = []
    for split in ["train", "val", "test"]:
        pool = split_pools[split]
        if not pool:
            print(f"  WARNING: No free intervals for {split} split negatives")
            continue

        # Weight intervals by length
        lengths = [end - start for _, _, start, end in pool]
        total_length = sum(lengths)
        if total_length == 0:
            continue

        n_target = targets[split]
        sampled = 0
        attempts = 0
        max_attempts = n_target * 10

        while sampled < n_target and attempts < max_attempts:
            attempts += 1
            # Pick a random interval (weighted by length)
            r = rng.random() * total_length
            cumsum = 0
            for i, length in enumerate(lengths):
                cumsum += length
                if cumsum >= r:
                    wav_start_t, wav_path, iv_start, iv_end = pool[i]
                    break

            # Pick a random offset within the interval
            max_offset = iv_end - CLIP_DURATION
            if max_offset <= iv_start:
                continue
            offset = rng.uniform(iv_start, max_offset)

            neg_utc = wav_start_t + __import__("datetime").timedelta(seconds=offset)
            negatives.append(
                {
                    "utc": neg_utc,
                    "wav_path": wav_path,
                    "offset_in_file": offset + CLIP_DURATION / 2,  # Center offset
                    "split": split,
                    "label": 0,
                }
            )
            sampled += 1

        print(f"  {split}: sampled {sampled}/{n_target} negatives")

    return negatives


# ════════════════════════════════════════════════════════════════════════
# Main processing pipeline
# ════════════════════════════════════════════════════════════════════════


def process_data():
    """Process annotations + WAVs → spectrograms + split files."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if all((DATA_SPLITS / f"{s}.pt").exists() for s in ["train", "val", "test"]):
        print("Data already processed. Delete data/splits/*.pt to reprocess.")
        return

    print("Loading annotations...")
    annotations = load_annotations(ANNOTATIONS_FILE)
    print(f"  {len(annotations)} confirmed Pv roar annotations")

    print("Building WAV index...")
    wav_index = build_wav_index(DATA_RAW)
    print(f"  {len(wav_index)} WAV files indexed")

    print("Processing positive clips...")
    rng = __import__("random").Random(RANDOM_SEED)

    # Process positives
    split_data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    skipped = 0
    for i, ann in enumerate(annotations):
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(annotations)}...")

        result = find_wav_for_time(wav_index, ann["utc"])
        if result is None:
            skipped += 1
            continue

        wav_start, wav_dur, wav_path, ann_offset = result
        # Center clip on annotation midpoint
        midpoint_offset = ann_offset + ann["duration"] / 2.0
        waveform = extract_clip(wav_path, midpoint_offset, wav_dur)
        if waveform is None:
            skipped += 1
            continue

        spec = compute_spectrogram(waveform)
        split = get_split(ann["utc"])
        split_data[split][0].append(spec)
        split_data[split][1].append(1)

    print(f"  Processed {len(annotations) - skipped} positives ({skipped} skipped)")

    print("Sampling negative clips...")
    negatives = sample_negatives(wav_index, annotations, rng)

    print("Processing negative clips...")
    neg_skipped = 0
    for i, neg in enumerate(negatives):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(negatives)}...")

        waveform = extract_clip(neg["wav_path"], neg["offset_in_file"],
                                torchaudio.info(neg["wav_path"]).num_frames / ORIG_SAMPLE_RATE)
        if waveform is None:
            neg_skipped += 1
            continue

        spec = compute_spectrogram(waveform)
        split = neg["split"]
        split_data[split][0].append(spec)
        split_data[split][1].append(0)

    print(f"  Processed {len(negatives) - neg_skipped} negatives ({neg_skipped} skipped)")

    # Save splits
    print("Saving splits...")
    metadata = {}
    for split in ["train", "val", "test"]:
        specs, labels = split_data[split]
        if not specs:
            print(f"  WARNING: {split} split is empty!")
            continue

        specs_tensor = torch.cat(specs, dim=0).unsqueeze(1) if specs[0].dim() == 2 else torch.stack(specs)
        # Ensure shape is (N, 1, N_MELS, T)
        if specs_tensor.dim() == 3:
            specs_tensor = specs_tensor.unsqueeze(1)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Shuffle deterministically
        perm = torch.randperm(len(labels_tensor), generator=torch.Generator().manual_seed(RANDOM_SEED))
        specs_tensor = specs_tensor[perm]
        labels_tensor = labels_tensor[perm]

        save_path = DATA_SPLITS / f"{split}.pt"
        torch.save({"spectrograms": specs_tensor, "labels": labels_tensor}, save_path)

        n_pos = int(labels_tensor.sum().item())
        n_neg = len(labels_tensor) - n_pos
        metadata[split] = {"n_samples": len(labels_tensor), "n_positive": n_pos, "n_negative": n_neg}
        print(f"  {split}: {n_pos} pos + {n_neg} neg = {len(labels_tensor)} total")
        print(f"    Spectrogram shape: {specs_tensor.shape}")

    # Save metadata
    with open(DATA_SPLITS / "metadata.json", "w") as f:
        json.dump(
            {
                "splits": metadata,
                "params": {
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "clip_duration": CLIP_DURATION,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "fmin": FMIN,
                    "fmax": FMAX,
                    "neg_pos_ratio": NEG_POS_RATIO,
                },
                "split_boundaries": {
                    "train": "Sep 1–20",
                    "val": "Sep 21–30",
                    "test": "Oct 1–15",
                },
            },
            f,
            indent=2,
        )

    print("Done!")


# ════════════════════════════════════════════════════════════════════════
# Dataset and evaluation (importable by train.py)
# ════════════════════════════════════════════════════════════════════════


class SealRoarDataset(Dataset):
    """PyTorch dataset for seal roar spectrograms."""

    def __init__(self, split_path):
        data = torch.load(split_path, weights_only=True)
        self.spectrograms = data["spectrograms"].float()
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


def load_splits():
    """Load train/val/test datasets. Returns (train_ds, val_ds, test_ds)."""
    return (
        SealRoarDataset(DATA_SPLITS / "train.pt"),
        SealRoarDataset(DATA_SPLITS / "val.pt"),
        SealRoarDataset(DATA_SPLITS / "test.pt"),
    )


def evaluate(model, dataset, device="cpu", batch_size=64):
    """Evaluate model on dataset. Returns metrics dict.

    Primary metric: val_f2 (F2 score — recall-weighted).
    The model should output raw logits (single value per sample).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            logits = model(specs)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # AUC-ROC (threshold-independent)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # Find threshold that maximizes F2 score
    best_f2 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (all_probs >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f2 = fbeta_score(all_labels, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = float(thresh)

    preds = (all_probs >= best_thresh).astype(int)

    return {
        "f2": float(best_f2),
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "auc_roc": float(auc),
        "precision": float(precision_score(all_labels, preds, zero_division=0)),
        "recall": float(recall_score(all_labels, preds, zero_division=0)),
        "threshold": best_thresh,
    }


# ════════════════════════════════════════════════════════════════════════
# Stats
# ════════════════════════════════════════════════════════════════════════


def show_stats():
    """Print dataset statistics."""
    if not (DATA_SPLITS / "metadata.json").exists():
        print("No processed data found. Run: uv run prepare.py")
        return

    with open(DATA_SPLITS / "metadata.json") as f:
        meta = json.load(f)

    print("\n=== Seal Roar Detection Dataset ===\n")
    print("Spectrogram parameters:")
    for k, v in meta["params"].items():
        print(f"  {k}: {v}")

    print("\nSplit boundaries:")
    for k, v in meta["split_boundaries"].items():
        print(f"  {k}: {v}")

    print("\nSplit statistics:")
    total = 0
    for split, stats in meta["splits"].items():
        n = stats["n_samples"]
        pos = stats["n_positive"]
        neg = stats["n_negative"]
        print(f"  {split:5s}: {pos:5d} pos + {neg:5d} neg = {n:5d} total  (pos ratio: {pos/n:.1%})")
        total += n
    print(f"  {'total':5s}: {total:5d}")

    # Show spectrogram shape if data exists
    for split in ["train", "val", "test"]:
        path = DATA_SPLITS / f"{split}.pt"
        if path.exists():
            data = torch.load(path, weights_only=True)
            print(f"\n  {split} spectrogram tensor: {data['spectrograms'].shape}")
            break


# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seal roar detection data pipeline")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        process_data()
