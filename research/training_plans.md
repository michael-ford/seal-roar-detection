# Training Plans

## Common to Both Plans

### Data Pipeline
- 3,461 confirmed "Pv roar" annotations (excluding 69 "Pv roar?")
- 4-second fixed windows centered on annotation midpoints
- Negative samples: random 4s windows from unannotated periods (with buffer around roars)
- Negative:positive ratio ~3:1 initially (tunable by agent)
- Temporal train/val/test split (Sep early → train, Sep late → val, Oct → test)
- Mel-spectrogram: 128 mel bands, 16 kHz, FFT 1024, hop 512, bandpass 50-4000 Hz, PCEN normalization

### Evaluation
- Primary metric: F1 score (recall-weighted, threshold tuned for high recall)
- Secondary: AUC-ROC, precision, recall reported separately
- Immutable eval function in prepare.py

---

## Plan A: Mac MPS Training

**Hardware**: Apple Silicon Mac, MPS backend (or CPU fallback)
**Constraint**: Limited VRAM (~16-24 GB unified memory shared with OS)

### Model Search Space
- **Baseline**: Custom lightweight CNN (3-5 conv layers, ~500K params)
- **Step up**: ResNet-18 (11M params) — should fit on MPS
- **Augmentations**: SpecAugment, time shift, random gain, mixup
- **Optimizer**: AdamW, cosine annealing
- **Loss**: Focal loss (gamma tunable, alpha biased toward recall)
- **Experiment time budget**: 2-3 minutes per run

### Pros
- Zero cost, immediate iteration
- Fast feedback loop for architecture/augmentation experiments
- Good enough for baseline development and augmentation search

### Cons
- Slower training than GPU, limits experiment throughput
- MPS has some PyTorch op coverage gaps (may need CPU fallback for some ops)
- Can't efficiently fine-tune large pretrained models (AST, PANNs, BEATs)
- Overnight autonomous run: ~150-200 experiments (at 3 min each, 10 hours)

### When to Use
- Initial development, debugging, pipeline validation
- Architecture search with small models
- Augmentation strategy experiments
- Any time you want fast iteration without cost

---

## Plan B: Cloud GPU Training

**Hardware**: Rented cloud GPU (recommendations below)
**Goal**: Larger models, pretrained backbones, higher experiment throughput

### Recommended Cloud Options (cost-effective)

| Provider | GPU | Cost/hr | Notes |
|----------|-----|---------|-------|
| **Vast.ai** | RTX 3090/4090 | $0.20-0.50/hr | Cheapest, community GPUs, good for batch jobs |
| **Lambda Labs** | A10G | $0.60/hr | Reliable, good PyTorch support |
| **RunPod** | RTX 4090 | $0.40-0.70/hr | Easy setup, persistent storage |
| **Google Colab Pro** | T4/A100 | $10/month | Easiest to start, less control |

**Estimated cost for a solid research run**: $5-20 total (10-40 GPU hours)

### Model Search Space (expanded)
- Everything from Plan A, plus:
- **ResNet-34/50** — deeper architectures
- **EfficientNet-B0/B2** — efficient scaling
- **PANNs CNN14** — pretrained on AudioSet (527 classes including animal sounds)
- **Audio Spectrogram Transformer (AST)** — ViT pretrained on AudioSet, fine-tune classification head
- **Experiment time budget**: 3-5 minutes per run (faster GPU = more experiments per hour)

### Pros
- 5-10x faster training per experiment
- Can fine-tune large pretrained models (AudioSet knowledge transfers well to bioacoustics)
- Overnight run: ~400-500 experiments (at 3 min each)
- Pretrained models likely give best absolute performance with limited data

### Cons
- Costs money (though modest)
- Setup overhead (SSH, data transfer, environment)
- Need to sync results back to local

### When to Use
- After Mac baseline is established and pipeline is validated
- When diminishing returns from small model experiments
- For final model training with best configuration found

---

## Recommended Workflow

```
Phase 1 (Mac): Pipeline + baseline
  → validate data loading, spectrogram generation, eval function
  → get a working CNN baseline with F1 > 0 (sanity check)
  → run overnight: augmentation + small architecture search

Phase 2 (Mac or GPU): Iterate
  → if Mac baseline is promising (F1 > 0.7), continue on Mac
  → if hitting ceiling, move to GPU for pretrained models

Phase 3 (GPU): Pretrained fine-tuning
  → PANNs or AST fine-tuning for best performance
  → overnight autonomous run with expanded search space
  → expected: F1 > 0.85 with good recall
```
