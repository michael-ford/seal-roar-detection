# Autoresearch — Framework Reference

## What It Is
Open-source project by Andrej Karpathy (March 2026) that lets an AI agent autonomously conduct ML research experiments. ~630 lines of Python across two files. The agent modifies code, runs time-boxed experiments, evaluates a single metric, logs results, keeps or reverts — repeat forever.

## Three-File Architecture (the pattern)
- **`prepare.py`** — Fixed infrastructure. Agent NEVER modifies. Data loading, preprocessing, evaluation function.
- **`train.py`** — Agent's search space. Model definition, optimizer, hyperparameters, training loop. Full freedom to modify.
- **`program.md`** — Human instructions for the agent. Experiment protocol, constraints, optimization target.

## Experiment Loop
1. Agent creates git branch, reads all files, creates `results.tsv`
2. **Cycle** (repeats forever):
   - Examine past results, form hypothesis
   - Modify `train.py`, commit
   - Run training (time-boxed, e.g. 5 min)
   - Parse output for metric + resource usage
   - Log to `results.tsv` with status: KEEP / DISCARD / CRASH
   - If improvement: keep commit. If regression/crash: `git reset`
3. Analysis notebook reads `results.tsv`, plots progress

## Key Design Decisions
- Single scalar metric as optimization target
- Time-boxed experiments for fair comparison
- Immutable evaluation function prevents gaming
- Git-based experiment tracking (branching + reset)
- Agent runs autonomously — "NEVER STOP"

## Dependencies (original)
PyTorch, kernels (FlashAttn3), rustbpe, tiktoken, pyarrow, numpy, pandas, matplotlib

## Adaptation for Audio Classification
### Keep
- Three-file architecture
- Experiment loop + results.tsv
- Git-based tracking
- analysis.ipynb for visualization
- uv for dependency management

### Replace
- `prepare.py`: Text/token pipeline → audio loading, spectrogram generation, PAMGuard annotation ingestion, clip extraction, audio-appropriate evaluation metric (F1, AUC-ROC, mAP)
- `train.py`: GPT model → audio classifier (ResNet/EfficientNet on spectrograms, or AST/PANNs fine-tuning). MuonAdamW → standard AdamW. CE loss → BCE/focal loss.
- `program.md`: LLM training instructions → seal roar detection domain knowledge, search space definition, augmentation hints
- `pyproject.toml`: NLP deps → torchaudio, librosa, soundfile, scikit-learn

### Time budget consideration
Original: 5 min on NVIDIA GPU for LLM training. For audio classification with smaller models on smaller datasets, 2-3 min may suffice. On macOS/MPS, adjust accordingly.
