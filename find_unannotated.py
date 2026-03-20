#!/usr/bin/env python3
"""Find model detections in unannotated regions for manual review.

Slides a 4-second window across WAV files, runs the trained model,
and saves clips where the model detects a roar but no annotation exists.
These are either false positives or missed annotations.

Usage:
    uv run find_unannotated.py                    # Process val-split WAV files
    uv run find_unannotated.py --split test       # Process test-split files
    uv run find_unannotated.py --max-clips 50     # Limit output clips
    uv run find_unannotated.py --threshold 0.5    # Custom detection threshold
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from prepare import (
    ANNOTATIONS_FILE,
    CLIP_DURATION,
    DATA_RAW,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    ORIG_SAMPLE_RATE,
    ROOT,
    TARGET_SAMPLE_RATE,
    TRAIN_END,
    VAL_END,
    apply_pcen,
    parse_wav_timestamp,
)
from train import SealRoarCNN, get_device

SLIDE_STEP = 2.0  # seconds — overlap between windows
OUTPUT_DIR = ROOT / "results" / "unannotated_detections"


def load_annotations_for_lookup():
    """Load annotations as list of (utc_start, utc_end) for overlap checking."""
    intervals = []
    with open(ANNOTATIONS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Label"].strip() != "Pv roar":
                continue
            utc = datetime.strptime(row["UTC"].strip(), "%Y-%m-%d %H:%M:%S.%f")
            dur = float(row["Duration"])
            intervals.append((utc, utc + timedelta(seconds=dur)))
    return intervals


def overlaps_annotation(window_start_utc, window_dur, annotations, buffer=2.0):
    """Check if a time window overlaps any annotation (with buffer)."""
    ws = window_start_utc - timedelta(seconds=buffer)
    we = window_start_utc + timedelta(seconds=window_dur + buffer)
    for ann_start, ann_end in annotations:
        if ws < ann_end and we > ann_start:
            return True
    return False


def get_split_for_time(utc):
    if utc < TRAIN_END:
        return "train"
    elif utc < VAL_END:
        return "val"
    else:
        return "test"


def process_wav_file(wav_path, model, device, mel_transform, resampler,
                     annotations, threshold, clips_found):
    """Slide window across a WAV file, find unannotated detections."""
    wav_start = parse_wav_timestamp(wav_path)
    info = sf.info(str(wav_path))
    wav_duration = info.duration
    detections = []

    offset = 0.0
    while offset + CLIP_DURATION <= wav_duration:
        window_utc = wav_start + timedelta(seconds=offset)

        # Skip if overlaps an annotation
        if overlaps_annotation(window_utc, CLIP_DURATION, annotations):
            offset += SLIDE_STEP
            continue

        # Extract and process clip
        frame_start = int(offset * ORIG_SAMPLE_RATE)
        num_frames = int(CLIP_DURATION * ORIG_SAMPLE_RATE)
        audio, sr = sf.read(str(wav_path), start=frame_start,
                           stop=frame_start + num_frames, dtype="float32")

        if len(audio) < num_frames * 0.9:
            offset += SLIDE_STEP
            continue
        if len(audio) < num_frames:
            audio = np.pad(audio, (0, num_frames - len(audio)))

        waveform = torch.from_numpy(audio).unsqueeze(0)
        waveform = resampler(waveform)
        mel_spec = mel_transform(waveform).unsqueeze(0)  # (1, 1, n_mels, time)
        spec = apply_pcen(mel_spec)

        # Run model
        with torch.no_grad():
            spec_dev = spec.to(device)
            logit = model(spec_dev).squeeze()
            prob = torch.sigmoid(logit).item()

        if prob >= threshold:
            detections.append({
                "wav_file": wav_path.name,
                "offset_sec": offset,
                "utc": window_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "probability": prob,
                "audio": audio,
            })

        offset += SLIDE_STEP

    return detections


def main():
    parser = argparse.ArgumentParser(description="Find unannotated model detections")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None,
                       help="Detection threshold (default: use model's optimal)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model_path = ROOT / "results" / "best_model.pt"
    if not model_path.exists():
        print("No trained model found. Run: uv run train.py")
        return

    model = SealRoarCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded")

    # Set threshold
    threshold = args.threshold if args.threshold is not None else 0.23  # baseline optimal
    print(f"Detection threshold: {threshold:.2f}")

    # Load annotations
    annotations = load_annotations_for_lookup()
    print(f"Loaded {len(annotations)} annotations")

    # Setup transforms
    import torchaudio
    resampler = torchaudio.transforms.Resample(ORIG_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
    )

    # Find WAV files for the requested split
    wav_files = sorted(DATA_RAW.glob("*.wav"))
    split_files = []
    for wf in wav_files:
        wt = parse_wav_timestamp(wf)
        if get_split_for_time(wt) == args.split:
            split_files.append(wf)
    print(f"Processing {len(split_files)} WAV files from {args.split} split...")

    # Process files
    all_detections = []
    for i, wf in enumerate(split_files):
        dets = process_wav_file(wf, model, device, mel_transform, resampler,
                               annotations, threshold, len(all_detections))
        all_detections.extend(dets)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(split_files)} files, {len(all_detections)} detections so far")
        if len(all_detections) >= args.max_clips:
            break

    # Sort by probability (most confident first)
    all_detections.sort(key=lambda d: d["probability"], reverse=True)
    all_detections = all_detections[:args.max_clips]

    # Save clips
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old clips
    for f in OUTPUT_DIR.glob("*.wav"):
        f.unlink()

    print(f"\nSaving {len(all_detections)} clips to {OUTPUT_DIR}/")
    print(f"{'#':>3}  {'Prob':>5}  {'UTC Time':>19}  {'WAV File'}")
    print("-" * 70)

    for i, det in enumerate(all_detections):
        # Save audio clip
        clip_name = f"{i+1:03d}_p{det['probability']:.2f}_{det['wav_file'].replace('.wav', '')}_{det['offset_sec']:.0f}s.wav"
        sf.write(str(OUTPUT_DIR / clip_name), det["audio"], ORIG_SAMPLE_RATE)
        print(f"{i+1:3d}  {det['probability']:.3f}  {det['utc']}  {det['wav_file']}")

    print(f"\nDone! Listen to clips in: {OUTPUT_DIR}/")
    print("If these sound like roars, they may be missed annotations (true positives).")


if __name__ == "__main__":
    main()
