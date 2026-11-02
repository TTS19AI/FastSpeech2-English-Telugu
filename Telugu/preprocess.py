
"""Manifest-driven raw feature extraction using the patched upstream extractor."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys

import numpy as np
import yaml
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    workflow = config["workflow"]
    sys.path.insert(0, workflow["reference_repo"])

    from preprocessor.preprocessor import Preprocessor

    manifest = (
        Path(workflow["manifest_directory"])
        / workflow[f"{args.split}_manifest"]
    )
    with manifest.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if args.limit:
        rows = rows[:args.limit]

    out = Path(config["path"]["preprocessed_path"])
    if args.limit:
        out = out / f"smoke_{args.split}_{args.limit}"

    # Never mix full-split outputs with this limited smoke test.
    local_config = copy.deepcopy(config)
    local_config["path"]["raw_path"] = str(
        Path(config["path"]["raw_path"]) / args.split
    )
    local_config["path"]["preprocessed_path"] = str(out)

    for feature in ("mel", "pitch", "energy", "duration", "TextGrid"):
        (out / feature).mkdir(parents=True, exist_ok=True)

    processor = Preprocessor(local_config)
    successful, rejected = [], []

    for row in tqdm(rows, desc=f"Extracting {args.split} raw features"):
        speaker = row["conditioning_group"]
        basename = f"telugu_{row['index']:06d}"
        paths = {
            feature: out / feature / f"{speaker}-{feature}-{basename}.npy"
            for feature in ("mel", "pitch", "energy", "duration")
        }

        try:
            destination = out / "TextGrid" / speaker / f"{basename}.TextGrid"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["textgrid_path"], destination)

            result = processor.process_utterance(speaker, basename)
            if result is None:
                raise ValueError("Upstream extractor rejected the recording")

            metadata, _, _, frames = result
            phones = metadata.split("|")[2].strip("{}").split()
            arrays = {
                feature: np.load(path) for feature, path in paths.items()
            }

            if arrays["mel"].shape != (frames, 80):
                raise ValueError("Unexpected mel shape")
            if int(arrays["duration"].sum()) != frames:
                raise ValueError("Duration sum does not equal mel length")
            if frames != row["duration_frames"]:
                raise ValueError("Frames differ from alignment audit")
            if len(arrays["duration"]) != len(phones):
                raise ValueError("Phone/duration count mismatch")
            if any(
                arrays[feature].shape != (len(phones),)
                for feature in ("pitch", "energy")
            ):
                raise ValueError("Phone-level feature shape mismatch")
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("Non-finite features")
            if np.any(arrays["duration"] < 0):
                raise ValueError("Negative duration")

            entry = dict(row)
            entry["upstream_metadata"] = metadata
            entry["feature_frames"] = int(frames)
            entry["feature_directory"] = str(out)
            successful.append(entry)

        except Exception as error:
            # Prevent partial arrays from being mistaken for successful output.
            for path in paths.values():
                if path.exists():
                    path.unlink()
            rejected.append({
                "audit_index": row["index"],
                "reason": f"{type(error).__name__}: {error}",
            })

    with (out / f"{args.split}_extracted.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for row in successful:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "split": args.split,
        "limited_smoke_test": bool(args.limit),
        "attempted_clips": len(rows),
        "successful_clips": len(successful),
        "rejected_clips": rejected,
        "source_hours_successful": sum(
            row["duration_seconds"] for row in successful
        ) / 3600,
        "feature_hours_successful": sum(
            row["feature_frames"] for row in successful
        ) * processor.hop_length / processor.sampling_rate / 3600,
        "features_normalized": False,
        "output_directory": str(out),
    }
    (out / f"{args.split}_extraction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)

    if rejected:
        raise SystemExit("Extraction has rejections; review before proceeding.")

if __name__ == "__main__":
    main()
