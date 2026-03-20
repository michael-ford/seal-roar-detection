#!/usr/bin/env python3
"""annotator.py — Gradio annotation review app for seal roar detections.

Usage:
    uv run annotator.py /path/to/wav/files/

Opens a browser tab where you can:
1. Scan a directory of WAV files with the trained model
2. Review each detection: spectrogram + audio + confidence
3. Label as Roar (TP) or Not Roar (FP)
4. Export confirmed roars to a Raven-compatible CSV
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt

# Import from project
from prepare import (
    CLIP_DURATION,
    FMAX,
    FMIN,
    ORIG_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    _get_mel_transform,
    _get_resampler,
    apply_pcen,
)
from train import SealRoarCNN

matplotlib.use("Agg")

# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

MODEL_PATH = Path(__file__).parent / "results" / "best_model.pt"
DETECTION_THRESHOLD = 0.40  # Optimal threshold from exp12
SLIDE_STEP = 1.0  # seconds between sliding windows
AMPLIFICATION = 200.0
NMS_WINDOW = 3.0  # seconds — suppress nearby detections within this range
DISPLAY_FMAX = 8000  # Hz — wider frequency range for display (capped at Nyquist for 16 kHz)

# ════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════


def load_model():
    """Load the trained SealRoarCNN model."""
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = SealRoarCNN().to(device)
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    )
    model.eval()
    return model, device


# ════════════════════════════════════════════════════════════════════════
# Audio processing
# ════════════════════════════════════════════════════════════════════════


def extract_clip_at(wav_path, start_sec, duration=CLIP_DURATION):
    """Extract a clip starting at start_sec from a WAV file.

    Returns (waveform_16khz, original_audio) or (None, None) on error.
    waveform_16khz: (1, n_samples) tensor at 16kHz for model input
    original_audio: raw numpy array at original sample rate for playback
    """
    info = sf.info(str(wav_path))
    sr = info.samplerate

    # Clamp to file bounds
    file_duration = info.duration
    start_sec = max(0.0, min(start_sec, file_duration - duration))

    frame_offset = int(start_sec * sr)
    num_frames = int(duration * sr)

    try:
        audio, file_sr = sf.read(
            str(wav_path), start=frame_offset, stop=frame_offset + num_frames,
            dtype="float32",
        )
    except Exception:
        return None, None

    expected = int(duration * sr)
    if len(audio) < expected * 0.9:
        return None, None
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))

    original_audio = audio.copy()
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)

    # Resample for model — handle files that aren't 36kHz
    if file_sr != ORIG_SAMPLE_RATE:
        resampler = __import__("torchaudio").transforms.Resample(file_sr, TARGET_SAMPLE_RATE)
    else:
        resampler = _get_resampler()
    waveform = resampler(waveform)

    return waveform, (original_audio, file_sr)


def compute_spectrogram(waveform):
    """Compute mel spectrogram + PCEN normalization."""
    mel_spec = _get_mel_transform()(waveform)
    return apply_pcen(mel_spec)


def bandpass_amplify(audio, sr, low=FMIN, high=FMAX, gain=AMPLIFICATION):
    """Bandpass filter and amplify audio for playback."""
    nyquist = sr / 2.0
    low_n = max(low / nyquist, 0.001)
    high_n = min(high / nyquist, 0.999)
    sos = butter(4, [low_n, high_n], btype="band", output="sos")
    filtered = sosfilt(sos, audio)
    amplified = filtered * gain
    amplified = np.clip(amplified, -1.0, 1.0)
    return amplified


# ════════════════════════════════════════════════════════════════════════
# Scanning
# ════════════════════════════════════════════════════════════════════════


def scan_wav_file(wav_path, model, device):
    """Scan a single WAV file with sliding window, return detections."""
    info = sf.info(str(wav_path))
    file_duration = info.duration
    detections = []

    if file_duration < CLIP_DURATION:
        return detections

    # Sliding window
    positions = np.arange(0, file_duration - CLIP_DURATION + 0.01, SLIDE_STEP)

    # Batch inference for speed
    batch_waveforms = []
    batch_positions = []

    for start in positions:
        waveform, _ = extract_clip_at(wav_path, start)
        if waveform is None:
            continue
        spec = compute_spectrogram(waveform)
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)  # (1, 1, 128, T)
        batch_waveforms.append(spec)
        batch_positions.append(start)

        # Process in batches of 32
        if len(batch_waveforms) >= 32:
            batch = torch.cat(batch_waveforms, dim=0).to(device)
            with torch.no_grad():
                logits = model(batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            for pos, prob in zip(batch_positions, probs):
                if prob >= DETECTION_THRESHOLD:
                    detections.append({
                        "file": str(wav_path),
                        "filename": Path(wav_path).name,
                        "start_time": float(pos),
                        "end_time": float(pos + CLIP_DURATION),
                        "confidence": float(prob),
                    })
            batch_waveforms = []
            batch_positions = []

    # Process remaining
    if batch_waveforms:
        batch = torch.cat(batch_waveforms, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        for pos, prob in zip(batch_positions, probs):
            if prob >= DETECTION_THRESHOLD:
                detections.append({
                    "file": str(wav_path),
                    "filename": Path(wav_path).name,
                    "start_time": float(pos),
                    "end_time": float(pos + CLIP_DURATION),
                    "confidence": float(prob),
                })

    return detections


def nms_detections(detections):
    """Non-maximum suppression: keep highest confidence within NMS_WINDOW."""
    if not detections:
        return []

    # Group by file
    by_file = {}
    for d in detections:
        by_file.setdefault(d["file"], []).append(d)

    kept = []
    for file_dets in by_file.values():
        # Sort by confidence descending
        file_dets.sort(key=lambda x: x["confidence"], reverse=True)
        selected = []
        for d in file_dets:
            # Check if too close to an already-selected detection
            suppressed = False
            for s in selected:
                if abs(d["start_time"] - s["start_time"]) < NMS_WINDOW:
                    suppressed = True
                    break
            if not suppressed:
                selected.append(d)
        kept.extend(selected)

    # Sort by confidence descending (highest first for review)
    kept.sort(key=lambda x: x["confidence"], reverse=True)
    return kept


# ════════════════════════════════════════════════════════════════════════
# Visualization
# ════════════════════════════════════════════════════════════════════════


_display_mel_transform = None


def _get_display_mel_transform():
    """Separate mel transform for display with wider frequency range (up to 9 kHz)."""
    global _display_mel_transform
    if _display_mel_transform is None:
        _display_mel_transform = __import__("torchaudio").transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=96,
            f_min=FMIN,
            f_max=DISPLAY_FMAX,
            power=2.0,
        )
    return _display_mel_transform


def make_spectrogram_image(wav_path, start_sec):
    """Generate a spectrogram image for a detection (display range up to 9 kHz)."""
    waveform, _ = extract_clip_at(wav_path, start_sec)
    if waveform is None:
        return None

    # Wider-range spectrogram for display
    mel_spec = _get_display_mel_transform()(waveform)
    spec = apply_pcen(mel_spec)
    spec_np = spec.squeeze().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5), dpi=120)
    ax.imshow(
        spec_np, aspect="auto", origin="lower", cmap="magma",
        extent=[0, CLIP_DURATION, FMIN / 1000, DISPLAY_FMAX / 1000],
    )
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("kHz", fontsize=9, rotation=0, labelpad=20)
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"{Path(wav_path).name} — {start_sec:.1f}s to {start_sec + CLIP_DURATION:.1f}s",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.18)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmpfile.name)
    plt.close(fig)
    return tmpfile.name


def make_playback_audio(wav_path, start_sec):
    """Generate bandpass-filtered, amplified audio for playback."""
    _, raw = extract_clip_at(wav_path, start_sec)
    if raw is None:
        return None

    audio, sr = raw
    processed = bandpass_amplify(audio, sr)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmpfile.name, processed, sr)
    return tmpfile.name


# ════════════════════════════════════════════════════════════════════════
# State management (auto-save)
# ════════════════════════════════════════════════════════════════════════


def get_save_path(wav_dir):
    """Get auto-save file path for a given WAV directory."""
    return Path(wav_dir) / ".annotator_state.json"


def save_state(wav_dir, detections, labels, current_idx):
    """Auto-save current review state."""
    state = {
        "detections": detections,
        "labels": labels,
        "current_idx": current_idx,
    }
    save_path = get_save_path(wav_dir)
    with open(save_path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(wav_dir):
    """Load saved state if it exists."""
    save_path = get_save_path(wav_dir)
    if save_path.exists():
        with open(save_path) as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════════════════════
# CSV export
# ════════════════════════════════════════════════════════════════════════


def export_csv(detections, labels, output_path):
    """Export reviewed detections to Raven-compatible CSV."""
    rows = []
    selection_num = 1
    for det, label in zip(detections, labels):
        if label is None:
            continue
        rows.append({
            "Selection": selection_num,
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin File": det["filename"],
            "Begin Time (s)": f"{det['start_time']:.3f}",
            "End Time (s)": f"{det['end_time']:.3f}",
            "Low Freq (Hz)": FMIN,
            "High Freq (Hz)": FMAX,
            "Confidence": f"{det['confidence']:.4f}",
            "Reviewer Label": label,
        })
        selection_num += 1

    import csv
    with open(output_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    return len(rows)


# ════════════════════════════════════════════════════════════════════════
# Gradio App
# ════════════════════════════════════════════════════════════════════════


def build_app(initial_dir=None):
    """Build the Gradio annotation review interface."""
    model, device = load_model()

    # Shared state
    app_state = {
        "detections": [],
        "labels": [],
        "current_idx": 0,
        "wav_dir": initial_dir or "",
        "scanning": False,
    }

    def scan_directory(wav_dir, progress=gr.Progress()):
        """Scan all WAV files in directory."""
        wav_dir = wav_dir.strip()
        if not wav_dir:
            return "Please enter a directory path.", "", None, None, gr.update(), gr.update()

        wav_path = Path(wav_dir)
        if not wav_path.is_dir():
            return f"Not a directory: {wav_dir}", "", None, None, gr.update(), gr.update()

        wav_files = sorted(wav_path.glob("*.wav")) + sorted(wav_path.glob("*.WAV"))
        if not wav_files:
            return "No WAV files found in directory.", "", None, None, gr.update(), gr.update()

        # Check for saved state
        saved = load_state(wav_dir)
        if saved:
            app_state["detections"] = saved["detections"]
            app_state["labels"] = saved["labels"]
            app_state["current_idx"] = saved["current_idx"]
            app_state["wav_dir"] = wav_dir
            n = len(saved["detections"])
            reviewed = sum(1 for l in saved["labels"] if l is not None)
            idx = saved["current_idx"]
            if n == 0:
                return f"Resumed session: 0 detections.", "", None, None, gr.update(), gr.update()
            det = app_state["detections"][idx]
            spec_img = make_spectrogram_image(det["file"], det["start_time"])
            audio_file = make_playback_audio(det["file"], det["start_time"])
            info = format_detection_info(det, idx, n, reviewed)
            status = f"Resumed session: {n} detections, {reviewed} already reviewed."
            return status, info, spec_img, audio_file, gr.update(interactive=True), gr.update(interactive=True)

        # Scan
        all_detections = []
        for i, wf in enumerate(wav_files):
            progress((i + 1) / len(wav_files), desc=f"Scanning {wf.name}...")
            dets = scan_wav_file(wf, model, device)
            all_detections.extend(dets)

        # NMS
        all_detections = nms_detections(all_detections)

        app_state["detections"] = all_detections
        app_state["labels"] = [None] * len(all_detections)
        app_state["current_idx"] = 0
        app_state["wav_dir"] = wav_dir

        n = len(all_detections)
        if n == 0:
            save_state(wav_dir, all_detections, app_state["labels"], 0)
            return f"Scan complete: {len(wav_files)} files, 0 detections.", "", None, None, gr.update(), gr.update()

        # Show first detection
        det = all_detections[0]
        spec_img = make_spectrogram_image(det["file"], det["start_time"])
        audio_file = make_playback_audio(det["file"], det["start_time"])
        info = format_detection_info(det, 0, n, 0)
        save_state(wav_dir, all_detections, app_state["labels"], 0)
        status = f"Scan complete: {len(wav_files)} files, {n} detections found."
        return status, info, spec_img, audio_file, gr.update(interactive=True), gr.update(interactive=True)

    def format_detection_info(det, idx, total, reviewed):
        """Format detection info for display."""
        label_str = ""
        if idx < len(app_state["labels"]) and app_state["labels"][idx] is not None:
            label_str = f"\n**Your label:** {app_state['labels'][idx]}"
        return (
            f"### Detection {idx + 1} of {total}  ·  Reviewed {reviewed}/{total}\n\n"
            f"**File:** {det['filename']}\n\n"
            f"**Time:** {det['start_time']:.1f}s – {det['end_time']:.1f}s\n\n"
            f"**Confidence:** {det['confidence']:.1%}"
            f"{label_str}"
        )

    def label_detection(label_value):
        """Apply a label and advance to next detection."""
        dets = app_state["detections"]
        if not dets:
            return "", None, None, gr.update(), gr.update()

        idx = app_state["current_idx"]
        app_state["labels"][idx] = label_value

        # Advance to next unlabeled, or next in sequence
        n = len(dets)
        reviewed = sum(1 for l in app_state["labels"] if l is not None)

        # Find next unlabeled
        next_idx = None
        for i in range(idx + 1, n):
            if app_state["labels"][i] is None:
                next_idx = i
                break
        if next_idx is None:
            # Wrap around
            for i in range(0, idx):
                if app_state["labels"][i] is None:
                    next_idx = i
                    break
        if next_idx is None:
            # All reviewed
            next_idx = idx  # Stay put

        app_state["current_idx"] = next_idx
        save_state(app_state["wav_dir"], dets, app_state["labels"], next_idx)

        if reviewed == n:
            info = f"### All {n} detections reviewed!\n\nClick **Export CSV** to save results."
            return info, None, None, gr.update(), gr.update()

        det = dets[next_idx]
        spec_img = make_spectrogram_image(det["file"], det["start_time"])
        audio_file = make_playback_audio(det["file"], det["start_time"])
        info = format_detection_info(det, next_idx, n, reviewed)
        return info, spec_img, audio_file, gr.update(interactive=True), gr.update(interactive=True)

    def nav(direction):
        """Navigate forward/backward through detections."""
        dets = app_state["detections"]
        if not dets:
            return "", None, None
        n = len(dets)
        idx = app_state["current_idx"]
        idx = (idx + direction) % n
        app_state["current_idx"] = idx
        reviewed = sum(1 for l in app_state["labels"] if l is not None)
        det = dets[idx]
        spec_img = make_spectrogram_image(det["file"], det["start_time"])
        audio_file = make_playback_audio(det["file"], det["start_time"])
        info = format_detection_info(det, idx, n, reviewed)
        return info, spec_img, audio_file

    def do_export():
        """Export reviewed labels to CSV."""
        dets = app_state["detections"]
        labels = app_state["labels"]
        if not dets:
            return "No detections to export."

        output_path = Path(app_state["wav_dir"]) / "reviewed_detections.tsv"
        n_exported = export_csv(dets, labels, output_path)
        return f"Exported {n_exported} reviewed detections to:\n`{output_path}`"

    # ── Build UI ──

    custom_css = """
    .spec-audio-stack .spec-img img {
        width: 100% !important;
        object-fit: fill !important;
    }
    .spec-audio-stack .spec-img {
        padding: 0 !important;
    }
    .spec-audio-stack .audio-player {
        padding-top: 0 !important;
    }
    """

    with gr.Blocks(
        title="Seal Roar Annotator",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:
        gr.Markdown("# Seal Roar Detection — Annotation Review")

        with gr.Row():
            wav_dir_input = gr.Textbox(
                label="WAV Directory",
                value=initial_dir or "",
                placeholder="/path/to/wav/files/",
                scale=4,
            )
            scan_btn = gr.Button("Scan Directory", variant="primary", scale=1)

        status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=3, elem_classes=["spec-audio-stack"]):
                detection_info = gr.Markdown("")
                spec_image = gr.Image(label="Spectrogram", type="filepath", elem_classes=["spec-img"], show_label=False)
                audio_player = gr.Audio(label="Audio (bandpass filtered, amplified 200x)", type="filepath", elem_classes=["audio-player"])

            with gr.Column(scale=1):
                gr.Markdown("### Label this detection")
                roar_btn = gr.Button("✓  Roar", variant="primary", size="lg", interactive=False)
                not_roar_btn = gr.Button("✗  Not Roar", variant="secondary", size="lg", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Navigate")
                with gr.Row():
                    prev_btn = gr.Button("← Prev")
                    next_btn = gr.Button("Next →")

                gr.Markdown("---")
                export_btn = gr.Button("Export CSV", variant="secondary")
                export_status = gr.Textbox(label="Export", interactive=False)

        # ── Wire events ──

        scan_btn.click(
            fn=scan_directory,
            inputs=[wav_dir_input],
            outputs=[status_text, detection_info, spec_image, audio_player, roar_btn, not_roar_btn],
        )

        roar_btn.click(
            fn=lambda: label_detection("Roar"),
            outputs=[detection_info, spec_image, audio_player, roar_btn, not_roar_btn],
        )
        not_roar_btn.click(
            fn=lambda: label_detection("Not Roar"),
            outputs=[detection_info, spec_image, audio_player, roar_btn, not_roar_btn],
        )

        prev_btn.click(
            fn=lambda: nav(-1),
            outputs=[detection_info, spec_image, audio_player],
        )
        next_btn.click(
            fn=lambda: nav(1),
            outputs=[detection_info, spec_image, audio_player],
        )

        export_btn.click(fn=do_export, outputs=[export_status])

    return app


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seal roar annotation review tool",
        usage="uv run annotator.py [/path/to/wav/files/]",
    )
    parser.add_argument("wav_dir", nargs="?", default=None, help="Directory containing WAV files to scan")
    args = parser.parse_args()

    app = build_app(initial_dir=args.wav_dir)
    app.launch(inbrowser=True)
