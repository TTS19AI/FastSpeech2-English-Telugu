
"""Plot saved predicted variance features for English or Telugu."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--language", choices=["English", "Telugu"], required=True)
args = parser.parse_args()

project = Path(__file__).resolve().parent.parent
results = project / "Results" / args.language
features = results / "variance_adaptor_predictions"
ids = (
    ["E1", "E2a", "E2b", "E3", "E4"]
    if args.language == "English"
    else ["T1", "T2a", "T2b", "T3", "T4"]
)
reports = []

for sentence_id in ids:
    record = json.loads(
        (features / f"{sentence_id}.json").read_text(encoding="utf-8")
    )
    with np.load(features / f"{sentence_id}.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}

    phones = record.get("model_phones", record.get("phones"))
    if not phones:
        raise RuntimeError(f"Missing phone labels: {sentence_id}")
    n = len(phones)

    for name in ("duration_frames", "duration_seconds", "pitch_hz", "energy_raw"):
        if arrays[name].shape != (n,) or not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"Invalid feature: {sentence_id}/{name}")

    duration = arrays["duration_frames"]
    if (duration < 0).any() or not np.all(duration == np.round(duration)):
        raise RuntimeError(f"Invalid duration values: {sentence_id}")
    if int(duration.sum()) != record["predicted_mel_frames"]:
        raise RuntimeError(f"Duration/mel mismatch: {sentence_id}")

    expected_seconds = (
        duration * record["hop_length"] / record["sample_rate"]
    )
    if not np.allclose(arrays["duration_seconds"], expected_seconds):
        raise RuntimeError(f"Duration conversion mismatch: {sentence_id}")

    audio_path = results / "test_generated_audios" / f"{sentence_id}.wav"
    audio = sf.info(audio_path)
    if audio.samplerate != record["sample_rate"] or audio.channels != 1:
        raise RuntimeError(f"Unexpected WAV format: {sentence_id}")
    if audio.frames != int(duration.sum()) * record["hop_length"]:
        raise RuntimeError(f"Duration/WAV length mismatch: {sentence_id}")

    stride = max(1, int(np.ceil(n / 32)))
    ticks = np.arange(0, n, stride)
    paths = {}

    for category, values, ylabel in [
        ("duration", arrays["duration_seconds"], "Predicted duration (seconds)"),
        ("pitch", arrays["pitch_hz"], "Inverse-normalized pitch (Hz)"),
        ("energy", arrays["energy_raw"], "Predicted energy (raw STFT scale)"),
    ]:
        folder = results / "variance_adaptor_plots" / category
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{sentence_id}.png"

        fig, ax = plt.subplots(figsize=(max(10, min(20, n * 0.18)), 4.5))
        if category == "duration":
            ax.bar(np.arange(n), values, width=0.8)
            ax.set_ylim(bottom=0)
        else:
            ax.plot(np.arange(n), values, marker=".", linewidth=1)

        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{i + 1}:{phones[i]}" for i in ticks],
            rotation=65, ha="right", fontsize=8,
        )
        ax.set_xlabel("Phoneme position : token (some tick labels omitted)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{sentence_id} — predicted {category} | "
            f"checkpoint {record['checkpoint_step']} | {record['speaker']}"
        )
        ax.grid(axis="y", alpha=0.3)

        notes = {
            "duration": (
                f"Frames × {record['hop_length']} / {record['sample_rate']}; "
                "predicted durations, not forced-alignment targets."
            ),
            "pitch": (
                "Phoneme-level F0 representation; training interpolated through "
                "unvoiced regions. Not a voiced/unvoiced contour."
            ),
            "energy": (
                "Inverse-normalized model prediction; not dB or physical intensity."
            ),
        }
        fig.text(0.01, 0.01, notes[category], fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[category] = str(path)

    reports.append({
        "id": sentence_id,
        "checkpoint_step": record["checkpoint_step"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "phone_count": n,
        "zero_duration_phones": int(np.count_nonzero(duration == 0)),
        "total_predicted_seconds": float(arrays["duration_seconds"].sum()),
        "pitch_hz_range": [
            float(arrays["pitch_hz"].min()), float(arrays["pitch_hz"].max())
        ],
        "energy_raw_range": [
            float(arrays["energy_raw"].min()), float(arrays["energy_raw"].max())
        ],
        "phone_labels": phones,
        "plots": paths,
    })
    print(f"[PASS] {sentence_id}: three plots saved; {n} phones", flush=True)

(results / "variance_plot_summary.json").write_text(
    json.dumps({
        "feature_source": "Model predictions, not features extracted from WAVs",
        "resolution": "phoneme",
        "plot_count": len(reports) * 3,
        "visual_verification_complete": False,
        "sentences": reports,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("Plots generated:", len(reports) * 3)
print("WAV/duration consistency: PASSED")
