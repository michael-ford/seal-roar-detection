# Seal Roar Detection — Annotation Review Tool

Browser-based tool for reviewing harbour seal (*Phoca vitulina*) roar detections in underwater acoustic recordings. A trained CNN scans WAV files and presents each detection for human review — spectrogram, amplified audio playback, and one-click labeling.

## Quick Start

### 1. Install uv (Python package manager)

**Mac:**
```bash
brew install uv
```

**Linux / WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Download this repository

```bash
git clone https://github.com/mikeford/seal-roar-detection.git
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
