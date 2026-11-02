
"""Generate Telugu speech and save traceable variance-adaptor predictions."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys

import numpy as np
import soundfile as sf
import torch
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=["T1", "T2a", "T2b", "T3", "T4"])
    parser.add_argument("--speaker", default="telugu_female")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--repo", default="/content/FastSpeech2_telugu")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    repository = Path(args.repo)
    sys.path.insert(0, str(repository))

    from text import text_to_sequence
    from text.symbols import symbols
    from utils.model import get_model, get_vocoder

    configs = tuple(
        yaml.safe_load(
            (project / "Telugu/config" / name).read_text(encoding="utf-8")
        )
        for name in ("preprocess.yaml", "model.yaml", "train.yaml")
    )
    preprocess, model_config, train = configs
    data = Path(preprocess["path"]["preprocessed_path"])
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    speakers = json.loads((data / "speakers.json").read_text(encoding="utf-8"))

    if args.speaker not in speakers:
        raise ValueError(f"Unknown speaker: {args.speaker}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu if intended.")
    device = torch.device(args.device)

    selection = json.loads(
        (project / "reproducibility/selected_checkpoints.json")
        .read_text(encoding="utf-8")
    )["Telugu"]
    step = selection["step"]
    checkpoint = Path(train["path"]["ckpt_path"]) / f"{step}.pth.tar"
    if checkpoint.resolve() != Path(selection["checkpoint"]).resolve():
        raise RuntimeError("Selected and configured checkpoint paths differ.")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != selection["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint hash mismatch.")

    entries = json.loads(
        (project / "Telugu/manifests/test_sentences_tokenized.json")
        .read_text(encoding="utf-8")
    )["sentences"]
    by_id = {entry["id"]: entry for entry in entries}
    if len(args.ids) != len(set(args.ids)) or any(i not in by_id for i in args.ids):
        raise ValueError("Unknown or duplicate sentence IDs.")

    output = project / "Results/Telugu"
    audio_dir = output / "test_generated_audios"
    feature_dir = output / "variance_adaptor_predictions"
    audio_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    for sentence_id in args.ids:
        for path in [
            audio_dir / f"{sentence_id}.wav",
            feature_dir / f"{sentence_id}.npz",
            feature_dir / f"{sentence_id}.json",
        ]:
            if path.exists():
                raise FileExistsError(f"Existing output preserved: {path}")

    symbol_ids = {symbol: i for i, symbol in enumerate(symbols)}
    model = get_model(
        SimpleNamespace(restore_step=step), configs, device, train=False
    )
    model.eval()
    vocoder = get_vocoder(model_config, device)
    sr = preprocess["preprocessing"]["audio"]["sampling_rate"]
    hop = preprocess["preprocessing"]["stft"]["hop_length"]

    for sentence_id in args.ids:
        entry = by_id[sentence_id]
        phones = entry["model_phones"]
        encoded = text_to_sequence(
            "{" + " ".join(phones) + "}",
            preprocess["preprocessing"]["text"]["text_cleaners"],
        )
        expected = [symbol_ids["@" + phone] for phone in phones]
        if encoded != expected or encoded != entry["token_ids"]:
            raise RuntimeError(f"Token encoding changed: {sentence_id}")

        tokens = torch.tensor([encoded], dtype=torch.long, device=device)
        lengths = torch.tensor([len(encoded)], dtype=torch.long, device=device)
        speaker = torch.tensor(
            [speakers[args.speaker]], dtype=torch.long, device=device
        )

        with torch.no_grad():
            prediction = model(speaker, tokens, lengths, len(encoded))
            frames = int(prediction[9][0].item())
            if not 0 < frames <= 4000:
                raise RuntimeError(f"Unexpected predicted length: {frames}")
            mel = prediction[1][0, :frames]
            if not torch.isfinite(mel).all().item():
                raise RuntimeError("Non-finite mel.")
            waveform = vocoder(mel.T.unsqueeze(0))[0, 0].cpu().numpy()

        n = len(encoded)
        duration = prediction[5][0, :n].detach().cpu().numpy()
        pitch = prediction[2][0, :n].detach().cpu().numpy()
        energy = prediction[3][0, :n].detach().cpu().numpy()

        if int(duration.sum()) != frames or (duration < 0).any():
            raise RuntimeError("Duration/mel inconsistency.")
        if not all(np.isfinite(a).all() for a in [waveform, pitch, energy]):
            raise RuntimeError("Non-finite predictions.")
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0
        if waveform.size == 0 or peak > 1:
            raise RuntimeError(f"Invalid waveform amplitude: {peak}")

        wav_path = audio_dir / f"{sentence_id}.wav"
        sf.write(wav_path, waveform, sr, subtype="PCM_16")
        np.savez(
            feature_dir / f"{sentence_id}.npz",
            duration_frames=duration,
            duration_seconds=duration * hop / sr,
            pitch_normalized=pitch,
            pitch_hz=pitch * stats["pitch"][3] + stats["pitch"][2],
            energy_normalized=energy,
            energy_raw=energy * stats["energy"][3] + stats["energy"][2],
        )

        record = {
            **entry,
            "checkpoint_step": step,
            "checkpoint_sha256": checkpoint_hash,
            "speaker": args.speaker,
            "speaker_id": speakers[args.speaker],
            "sample_rate": sr,
            "hop_length": hop,
            "predicted_mel_frames": frames,
            "duration_seconds": waveform.size / sr,
            "waveform_peak": peak,
            "wav_path": str(wav_path),
            "features": "Predicted phoneme-level variance-adaptor outputs",
            "pitch_note": "Inverse-normalized F0 target representation; training used interpolation through unvoiced regions.",
            "energy_note": "Raw STFT energy scale; not dB or physical intensity.",
            "listening_review_complete": False,
        }
        (feature_dir / f"{sentence_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({
            "id": sentence_id, "checkpoint": step,
            "speaker": args.speaker,
            "duration_seconds": record["duration_seconds"],
            "wav_path": str(wav_path),
            "variance_arrays_saved": True,
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
