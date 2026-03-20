# Seal Roar Detection — Annotation Review Tool

Browser-based tool for reviewing harbour seal (*Phoca vitulina*) roar detections in underwater acoustic recordings. A trained CNN scans WAV files and presents each detection for human review — spectrogram, amplified audio playback, and one-click labeling.

## Quick Start

### 1. Install uv (Python package manager)

**Mac / Linux / WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

No Homebrew, pip, or conda needed — just this one command.

### 2. Download this repository

```bash
git clone https://github.com/michael-ford/seal-roar-detection.git
cd seal-roar-detection
```

Or download as a ZIP from the green "Code" button on GitHub and unzip it.

### 3. Run the annotator

```bash
uv run annotator.py /path/to/your/wav/files/
```

That's it. On first run, `uv` automatically installs Python and all dependencies — no manual setup needed. A browser tab opens with the annotation interface.

## How It Works

1. **Scan** — Point the app at a directory of WAV files. The model scans each file with a sliding 4-second window and finds potential roars.

2. **Review** — Detections are shown highest-confidence first. For each one you see:
   - Spectrogram (50 Hz – 8 kHz)
   - Audio player (bandpass filtered 50–4000 Hz, amplified 200x so roars are audible)
   - Confidence score
   - Timestamp in the original file

3. **Label** — Click **Roar** or **Not Roar** for each detection. Progress is auto-saved after every click — you can close the tab and resume later.

4. **Export** — Click **Export CSV** to save a `reviewed_detections.tsv` in the WAV directory. The file is tab-separated and compatible with Raven selection tables.

## Export Format

| Column | Description |
|---|---|
| Selection | Sequential number |
| View | Spectrogram 1 |
| Channel | 1 |
| Begin File | WAV filename |
| Begin Time (s) | Detection start in file |
| End Time (s) | Detection end in file |
| Low Freq (Hz) | 50 |
| High Freq (Hz) | 4000 |
| Confidence | Model confidence (0–1) |
| Reviewer Label | Roar / Not Roar |

## Requirements

- **OS:** Mac, Linux, or Windows
- **Disk:** ~500 MB (for Python + dependencies, downloaded automatically)
- **RAM:** 4 GB minimum
- **Python:** Handled automatically by `uv` — you don't need to install Python yourself

## Troubleshooting

**"command not found: uv"** — Restart your terminal after installing uv, or add it to your PATH as shown in the uv installer output.

**Slow scanning** — Each WAV file takes a few seconds. The progress bar shows which file is being scanned. You can start reviewing detections from previous scans while new files are still scanning by resuming a saved session.

**No detections found** — The model may not detect roars if the recordings don't contain harbour seal vocalizations, or if the audio quality differs significantly from the training data (SoundTrap 36 kHz recordings from Lasqueti Island, BC).

**Audio sounds wrong** — The playback is bandpass filtered (50–4000 Hz) and amplified 200x. This is intentional — harbour seal roars are very quiet in raw underwater recordings.

## Model Development

### Data

- **Source:** SoundTrap ST303 hydrophone recordings from Lasqueti Island, BC (Sep–Oct 2017)
- **Annotations:** 3,461 confirmed harbour seal roars from PAMGuard spectrogram annotations
- **Audio:** 36 kHz, mono, 16-bit WAV (5-minute recordings)
- **Features:** 128-band mel spectrograms, PCEN normalization, bandpass 50–4000 Hz, resampled to 16 kHz
- **Clip duration:** 4 seconds
- **Negative class:** Background noise including orca calls, ambient ocean noise, boat noise (3:1 neg:pos ratio)

### Train / Val / Test Split

Temporal split to test generalization across time:

| Split | Dates | Roars | Background | Total |
|-------|-------|-------|------------|-------|
| Train | Sep 1, 2017 | 1,745 | 5,235 | 6,980 |
| Val | Sep 2, 2017 | 782 | 2,346 | 3,128 |
| Test | Oct 1–15, 2017 | 934 | 2,802 | 3,736 |

The test set is one month later than training data, testing temporal generalization (different noise conditions, potentially different individual seals).

### Experiment Results

18 experiments were run using the [autoresearch](https://github.com/karpathy/autoresearch) pattern — autonomous ML experimentation with time-boxed training, single-metric optimization (F2 score, recall-weighted), and git-based experiment tracking.

**Top models (sorted by val F2):**

| Experiment | Model | F2 | Recall | Precision | AUC | Params |
|-----------|-------|-----|--------|-----------|-----|--------|
| gpu04 | ResNet-18 + label smoothing | **0.920** | 96.4% | 77.6% | 0.980 | 11.2M |
| gpu01 | ResNet-18 pretrained | 0.918 | 96.0% | 78.0% | 0.975 | 11.2M |
| gpu02 | ResNet-18, lower LR | 0.915 | 97.1% | 74.3% | 0.977 | 11.2M |
| gpu05 | PANNs CNN14 (AudioSet) | 0.902 | 95.0% | 74.8% | 0.974 | 79.7M |
| exp02 | CNN + residual blocks | 0.903 | 94.8% | 75.9% | 0.973 | 777K |
| **exp12** | **Depthwise-sep CNN** | **0.901** | **96.3%** | **71.7%** | **0.972** | **47K** |
| baseline | 4-layer CNN | 0.891 | 95.0% | 71.4% | 0.969 | 389K |

**Deployed model:** exp12 (depthwise-separable CNN, 47K params). Chosen over the GPU models because:
- Nearly identical recall (96.3% vs 96.4%)
- 240x smaller (47K vs 11.2M params)
- ~50x faster inference — important for scanning hours of audio
- No torchvision dependency

### Cross-Validation

3-fold temporal cross-validation rotating Sep 1 / Sep 2 / Oct 1–15 through train/val/test:

**Depthwise-sep CNN (deployed model, 47K params):**

| Metric | Mean ± Std | Fold 1 | Fold 2 | Fold 3 |
|--------|-----------|--------|--------|--------|
| Test F2 | 0.882 ± 0.019 | 0.855 | 0.897 | 0.895 |
| Test Recall | 0.953 ± 0.005 | 0.949 | 0.951 | 0.959 |
| Test Precision | 0.684 ± 0.051 | 0.613 | 0.731 | 0.708 |
| Test AUC | 0.963 ± 0.008 | 0.951 | 0.969 | 0.968 |

**ResNet-18 pretrained (GPU model, 11.2M params):**

| Metric | Mean ± Std | Fold 1 | Fold 2 | Fold 3 |
|--------|-----------|--------|--------|--------|
| Test F2 | **0.908 ± 0.007** | 0.902 | 0.904 | 0.919 |
| Test Recall | **0.958 ± 0.017** | 0.953 | 0.939 | 0.981 |
| Test Precision | **0.754 ± 0.022** | 0.744 | 0.785 | 0.733 |
| Test AUC | **0.975 ± 0.004** | 0.972 | 0.972 | 0.980 |

ResNet-18 is more consistent across folds (lower std) and has ~3% higher F2. Both models maintain >93% recall across all folds.

**PANNs CNN14 (pretrained on AudioSet)** scored 0.902 F2 on the default split — comparable to the lightweight model despite having 80M parameters. AudioSet pretraining did not provide meaningful advantage over ImageNet pretraining for this task, likely due to the small dataset size (3,461 labeled roars).

### False Positive Analysis

Manual review of the top 15 model detections in unannotated regions found that **8 of 15 (53%) were actual roars** missed during annotation. This means the true precision is significantly higher than measured (~88–92% estimated vs 72% measured), and the annotations have incomplete coverage.

### Full Experiment Log

All 18 experiments are logged in [`results.tsv`](results.tsv).

| Tag | Status | F2 | Description |
|-----|--------|-----|-------------|
| baseline | KEEP | 0.891 | 4-layer CNN, focal loss, AdamW lr=1e-3 |
| exp01 | KEEP | 0.899 | + SpecAugment |
| exp02 | KEEP | 0.903 | + Residual connections (777K params) |
| exp03 | DISCARD | 0.891 | OneCycleLR lr=3e-3 |
| exp04 | DISCARD | 0.894 | Mixup alpha=0.3 |
| exp05 | DISCARD | 0.877 | Lower dropout, higher weight decay |
| exp06 | DISCARD | 0.861 | Batch=64, lr=2e-3, 5min budget |
| exp07 | DISCARD | 0.886 | 5min budget at lr=1e-3 |
| exp08 | DISCARD | 0.862 | SE attention blocks |
| exp09 | DISCARD | 0.890 | Gradient clipping + validate every epoch |
| exp10 | DISCARD | 0.852 | Wider channels (3M params, too slow on MPS) |
| exp11 | DISCARD | 0.875 | Focal alpha=0.85, gamma=1.0 |
| exp12 | KEEP | 0.901 | Depthwise-separable CNN (47K params) |
| gpu01 | KEEP | 0.918 | ResNet-18 pretrained (Vast.ai RTX 3060) |
| gpu02 | KEEP | 0.915 | ResNet-18 lower LR, 5min |
| gpu03 | DISCARD | 0.890 | ResNet-18 discriminative LR |
| gpu04 | KEEP | 0.920 | ResNet-18 + label smoothing + stronger aug |
| gpu05 | KEEP | 0.902 | PANNs CNN14 pretrained on AudioSet |
