# Seal Roar Detection — Agent Instructions

You are autonomously optimizing a binary classifier that detects harbour seal (*Phoca vitulina*) roars in underwater acoustic recordings from Lasqueti Island, BC.

## Architecture

- **`prepare.py`** — LOCKED. Data pipeline, spectrogram generation, evaluation function. **Do NOT modify.**
- **`train.py`** — YOUR SEARCH SPACE. Modify freely: model architecture, optimizer, hyperparameters, augmentations, loss function, training loop.
- **`results.tsv`** — Experiment log. Append results; never delete rows.

## Optimization Target

**Maximize `val_f2`** — the F2 score on the validation set (recall weighted 4× over precision).

The user cares more about recall (don't miss roars) than precision (some false alarms are acceptable).

## Experiment Protocol

1. Read `results.tsv` to understand what has been tried and what worked.
2. Read `train.py` to understand the current state.
3. Form a hypothesis about what change will improve `val_f2`.
4. Modify `train.py` — make ONE focused change per experiment.
5. Commit with a descriptive message: `git commit -am "experiment: <description>"`.
6. Run: `uv run train.py > run.log 2>&1`
7. Parse `run.log` for `val_f2=`, `val_f1=`, `val_auc_roc=`, `val_precision=`, `val_recall=`, `val_threshold=`, `train_time=`, `n_params=`.
8. Log results to `results.tsv`:
   - **KEEP**: `val_f2` improved over previous best → keep the commit
   - **DISCARD**: `val_f2` did not improve → `git reset --hard HEAD~1`
   - **CRASH**: training failed → `git reset --hard HEAD~1`, note the error
9. **Go to step 1. NEVER STOP.**

## Data Details

- **Spectrograms**: (1, 128, 126) — 128 mel bands × 126 time frames, PCEN-normalized
- **Sample rate**: 16 kHz (resampled from 36 kHz SoundTrap recordings)
- **Clip duration**: 4 seconds
- **Frequency range**: 50–4000 Hz (mel filterbank bounds)
- **Positive class**: harbour seal roar (3,461 annotations, mean duration 2.8s)
- **Negative class**: background (includes orca calls, ambient noise, boat noise)
- **Neg:Pos ratio**: 3:1 in all splits
- **Splits**: temporal — Train: Sep 1–20, Val: Sep 21–30, Test: Oct 1–15

## Domain Knowledge

- Harbour seal roars: broadband, 50–2000 Hz primary energy, 1–5 second duration
- Roars are amplitude-modulated with pulsatile structure, individually distinctive
- Orca calls overlap in frequency range but have different temporal/harmonic structure
- Underwater recordings have highly variable SNR (-10 to +10 dB)
- PCEN normalization already handles some noise variation

## Search Space Ideas

### Architecture (small → large)
- Simple CNN (current baseline: ~250K params)
- Deeper CNN with residual connections
- ResNet-18 (modify first conv for 1-channel input)
- EfficientNet-B0 with 1-channel adapter
- Attention mechanisms (SE blocks, CBAM)

### Augmentation (applied in train.py, NOT in prepare.py)
- **SpecAugment**: random time/frequency masking on spectrograms
- **Mixup**: blend spectrograms and labels
- **Time shift**: roll spectrogram along time axis
- **Random gain**: scale spectrogram amplitude
- **Gaussian noise**: add noise to spectrograms
- **Cutout**: mask random rectangular regions

### Loss & Class Balance
- Focal loss (tune alpha, gamma)
- Class-weighted BCE
- Label smoothing
- Asymmetric loss (penalize FN more than FP)

### Optimizer & Schedule
- AdamW (tune lr, weight_decay)
- SGD with momentum
- OneCycleLR, ReduceLROnPlateau
- Warmup + cosine decay
- Gradient clipping

### Regularization
- Dropout rate
- Weight decay
- Early stopping patience
- Batch size effects

## Constraints

- **Time budget**: each experiment must complete within the TIME_BUDGET set in train.py
- **Device**: use `get_device()` — supports MPS (Mac), CUDA, and CPU
- **Memory**: keep peak memory reasonable for Mac (16 GB unified memory)
- **Dependencies**: only use packages in `pyproject.toml` — do NOT add new dependencies
- **Evaluation**: always use `from prepare import evaluate` — do NOT implement your own metrics

## Results Format

Tab-separated in `results.tsv`:
```
tag	status	val_f2	val_f1	val_auc_roc	val_precision	val_recall	val_threshold	train_time	n_params	description
```
