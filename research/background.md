# Seal Roar Detection — Background Research

## Domain: Pinniped Underwater Acoustics

### Species & Acoustic Characteristics
- **Northern Elephant Seal**: Pulsed roars, fundamental 200-900 Hz, energy <3 kHz, 1-4s duration, individually distinctive rhythm
- **Southern Elephant Seal**: Fundamental 100-500 Hz, broadband to 3-4 kHz, "gurgling roar" amplified by proboscis
- **Grey Seal**: Broadband 100 Hz - 5+ kHz, most energy <2 kHz, 0.5-3s, underwater "rups" and rumbles
- **Weddell Seal**: 34+ call types, 200 Hz - 12+ kHz, 0.2-6s, under-ice high amplitude
- **Leopard Seal**: Low-frequency broadcast calls 200-1000 Hz, some to 6 kHz
- **Bearded Seal**: Long descending trills 3-4 kHz → 300 Hz, 30-70s duration

### General Roar Characteristics
- Low-to-mid frequency (fundamental 100-1000 Hz)
- Broadband harmonic structure, energy up to 3-8 kHz
- High amplitude, often pulsatile/amplitude-modulated
- Sexually dimorphic (predominantly male)
- Context: agonistic, territorial, breeding

## Key Challenges
1. **Noise**: Ambient ocean, anthropogenic (ships, sonar), equipment self-noise. SNR often -10 to +10 dB
2. **Class imbalance**: Target calls <1% of recording time; 1000:1 background-to-call ratios common
3. **Variable conditions**: Hydrophone depth, water properties, equipment differences, seasonal/diel variation
4. **Generalization**: Models trained on one site often fail at new sites (domain shift)
5. **Labeling cost**: Expert annotation is slow and expensive; inter-annotator agreement can be low
6. **Overlapping calls**: Multi-species environments, chorus events

## State of the Art — ML for Marine Bioacoustics
- **Dominant approach**: CNNs on mel-spectrograms (ResNet, EfficientNet, VGG)
- **Pretrained models**: PANNs (CNN14 on AudioSet), AST, BEATs, HuBERT, wav2vec 2.0
- **Emerging**: Audio Spectrogram Transformer, self-supervised learning, few-shot/prototypical networks
- **YOLO-style detection**: Treating calls as objects in spectrograms
- **Production systems**: PAMGuard, JASCO, Wildlife Acoustics Kaleidoscope

## Recommended Preprocessing
- Sample rate: 16 kHz (sufficient for energy below 8 kHz)
- Bandpass: 50-5000 Hz
- Mel-spectrogram: 128 mel bands, FFT 1024-2048, hop 512
- **PCEN normalization** (handles variable recording conditions — strongly preferred over log-mel)
- Window size: 3-5 seconds (captures full roar duration)

## Key References
- Shiu et al. (2020) — Deep neural networks for marine mammal detection
- Bergler et al. (2019) — ORCA-SPOT CNN pipeline
- Kirsebom et al. (2020) — ResNet right whale upcalls, generalization evaluation
- Stowell (2022) — Computational bioacoustics review & roadmap
- Gong et al. (2021) — Audio Spectrogram Transformer

## Relevant Open-Source Tools
- **Ketos** — Python ML for underwater acoustics (spectrograms + CNN training)
- **OpenSoundscape** — Bioacoustic analysis with CNN detection
- **PAMGuard** — PAM platform, annotation, detector plugins
- **ANIMAL-SPOT** — General marine mammal detection
- **Koogu** — Training/deploying bioacoustic classifiers
