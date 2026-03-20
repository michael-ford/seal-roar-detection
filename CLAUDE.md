# Seal Roar Detection

Underwater acoustic detection model for seal roars, using the autoresearch pattern (Karpathy) for autonomous ML experimentation.

## Project Structure
```
seal-roar-detection/
  prepare.py          # LOCKED — data pipeline, eval function (agent cannot modify)
  train.py            # SEARCH SPACE — model, optimizer, hyperparams (agent modifies)
  program.md          # Agent instructions
  analysis.ipynb      # Experiment visualization
  results.tsv         # Experiment log
  pyproject.toml      # Dependencies (uv)
  data/
    raw/              # Original .wav recordings + PAMGuard files
    processed/        # Generated spectrograms, cached features
    splits/           # Train/val/test manifests
  annotations/        # PAMGuard exports, Raven selection tables
  research/           # Background research docs
  results/            # Saved model checkpoints, analysis outputs
```

## Key Decisions
- **Species**: Harbour seal (Phoca vitulina), Lasqueti Island BC
- **Task**: Binary classification — "Pv roar" vs background (background includes orca calls, ambient noise)
- **Labels**: Only confirmed "Pv roar" annotations used (3,461). "Pv roar?" excluded.
- **Window**: Fixed 4-second clips
- **Audio**: Mel-spectrograms with PCEN normalization, 128 mel bands, 16 kHz resample, bandpass 50-4000 Hz
- **Evaluation**: F1 score (recall-weighted), AUC-ROC
- **Training**: Two configurations — Mac MPS (lightweight models) and cloud GPU (larger models/pretrained)
- **Annotations**: CSV export from PAMGuard spectrogram annotations (not SQLite)

## Commands
- `uv run prepare.py` — process annotations + WAVs, generate spectrograms, build splits
- `uv run train.py` — run a single training experiment
- Uses `uv` for all dependency management (never pip)
