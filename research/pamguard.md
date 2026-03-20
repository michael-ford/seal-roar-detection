# PAMGuard — Data Format Reference

## Overview
PAMGuard (Passive Acoustic Monitoring Guard) is open-source Java software for detecting, classifying, and localizing marine mammal vocalizations. Maintained by SMRU at University of St Andrews.

## Storage Formats

### 1. Database (.sqlite3)
- Default backend since PAMGuard 2.x (older: MS Access .mdb)
- Stores detection metadata, annotations, settings
- Each module creates its own table(s)
- **Key tables**:
  - `Click_Detector_Clicks` — click detections
  - `Spectrogram_Annotation` — manual spectrogram annotations
  - `Deep_Learning_Detection` — DL classifier outputs
  - `GPS_Data`, `Sound_Acquisition` — context metadata

### 2. Binary Store (.pgdf)
- PAMGuard Data Files — custom binary format
- Stores full detection data (waveforms, spectrogram snippets, whistle contours)
- Organized by date, chunked by time period
- Naming: `ModuleName_Date_Time.pgdf`
- Read with: PAMBinaries Python library

### 3. Audio Files (.wav, .aif, .flac)
- Standard audio formats

## Annotation Fields (typical)
```
UID             — unique identifier (int64)
UTC             — start time (ms precision)
duration        — milliseconds or samples
StartSample     — sample number in audio stream
ChannelBitmap   — which channels
FreqLimits      — low/high freq (Hz)
Amplitude       — peak amplitude (dB)
species         — species label
callType        — call type label
Comment         — analyst notes
BinaryFile      — path to .pgdf with waveform
```

## Spectrogram Annotation Table
```
Id, UID, UTC, UTCMilliseconds
Duration (seconds)
f1 (low frequency Hz)
f2 (high frequency Hz)
Label / Species
Note / Comment
Channel
```

## Reading PAMGuard Data in Python

### SQLite + pandas (fastest path to training data)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('pamguard_data.sqlite3')
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
annotations = pd.read_sql("SELECT * FROM Spectrogram_Annotation", conn)
```

### Clip extraction from timestamps
```python
import soundfile as sf
# Given detection at time T with duration D:
audio, sr = sf.read('recording.wav', start=int(T*sr), stop=int((T+D)*sr))
```

### Binary files
- Use `PAMBinaries` Python library (github.com/PAMGuard/PAMBinaries)

### Raven Selection Tables (.txt export)
Tab-delimited: Selection, View, Channel, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Annotation
```python
raven = pd.read_csv('selections.txt', sep='\t')
```

## Recommended ML Pipeline
1. Query SQLite database for annotation timestamps + labels
2. Extract audio clips from .wav files using timestamps
3. Generate spectrograms from clips
4. (Optional) Parse .pgdf for pre-extracted waveform snippets
