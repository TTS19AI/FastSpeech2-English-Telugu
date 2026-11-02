from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = config_path.parent.parent
    records = project / "reproducibility"

    fit = json.loads(
        (records / "english_training_normalization_statistics.json")
        .read_text(encoding="utf-8")
    )
    if fit["fit_split"] != "train":
        raise ValueError("Expected training-only statistics.")

    raw = Path(config["path"]["preprocessed_path"])
    output = raw.parent / "english_processed"
    if output.resolve() == raw.resolve():
        raise ValueError("Processed output must differ from raw input.")

    manifests = {}
    for split in ("train", "validation"):
        with (raw / f"{split}_extracted.jsonl").open(encoding="utf-8") as f:
            manifests[split] = [
                json.loads(line) for line in f if line.strip()
            ]

    if len(manifests["train"]) != 3135 or len(manifests["validation"]) != 626:
        raise ValueError("Unexpected extracted counts.")

    speakers = {
        name: index for index, name in enumerate(sorted({
            row["conditioning_group"] for row in manifests["train"]
        }))
    }
    if any(
        row["conditioning_group"] not in speakers
        for row in manifests["validation"]
    ):
        raise ValueError("Validation conditioning group missing from training.")

    for feature in ("mel", "duration", "pitch", "energy"):
        (output / feature).mkdir(parents=True, exist_ok=True)

    bounds = {
        feature: [float("inf"), float("-inf")]
        for feature in ("pitch", "energy")
    }
    summary = {}

    for split, rows in manifests.items():
        metadata = []
        zero_positions = 0

        for row in rows:
            speaker = row["conditioning_group"]
            basename = f"svarah_{row['index']:06d}"

            def path(feature, root):
                return root / feature / f"{speaker}-{feature}-{basename}.npy"

            duration = np.load(path("duration", raw))
            valid = duration > 0
            zero_positions += int(np.count_nonzero(~valid))

            for feature in ("mel", "duration"):
                shutil.copyfile(path(feature, raw), path(feature, output))

            for feature in ("pitch", "energy"):
                values = np.load(path(feature, raw))
                if values.shape != duration.shape:
                    raise ValueError(f"Shape mismatch: {basename}/{feature}")

                stats = fit["statistics"][feature]
                normalized = (
                    (values.astype(np.float64) - stats["mean"]) / stats["std"]
                )
                normalized[~valid] = 0.0
                normalized = normalized.astype(np.float32)

                if not np.isfinite(normalized).all():
                    raise ValueError(f"Non-finite values: {basename}/{feature}")

                np.save(path(feature, output), normalized)

                # Embedding-bin bounds are fitted on training data only.
                if split == "train" and np.any(valid):
                    bounds[feature][0] = min(
                        bounds[feature][0], float(normalized[valid].min())
                    )
                    bounds[feature][1] = max(
                        bounds[feature][1], float(normalized[valid].max())
                    )

            parts = row["upstream_metadata"].split("|", 3)
            if len(parts) != 4:
                raise ValueError(f"Invalid upstream metadata: {basename}")

            original = row["text"]
            if "|" in original:
                raise ValueError(f"Metadata pipe in transcript: {basename}")
            # Metadata must occupy one line. The JSON source stays unchanged.
            original = original.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

            metadata.append(
                "|".join([basename, speaker, parts[2], original])
            )

        filename = "train.txt" if split == "train" else "val.txt"
        (output / filename).write_text(
            "\n".join(metadata) + "\n", encoding="utf-8"
        )
        summary[split] = {
            "clips": len(rows),
            "zero_duration_positions_set_to_zero": zero_positions,
        }

    stats_json = {}
    for feature in ("pitch", "energy"):
        lower, upper = bounds[feature]
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid training bounds for {feature}")
        stats = fit["statistics"][feature]
        stats_json[feature] = [
            lower, upper, stats["mean"], stats["std"]
        ]

    (output / "stats.json").write_text(
        json.dumps(stats_json, indent=2), encoding="utf-8"
    )
    (output / "speakers.json").write_text(
        json.dumps(speakers, indent=2), encoding="utf-8"
    )

    report = {
        "output_directory": str(output),
        "splits": summary,
        "conditioning_groups": len(speakers),
        "statistics_and_bin_bounds_fit_split": "train",
        "validation_values_clipped": False,
        "zero_duration_normalized_value": 0.0,
        "raw_arrays_modified": False,
        "stats": stats_json,
    }
    (records / "english_normalization_run.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("English normalization and metadata creation: PASSED")

if __name__ == "__main__":
    main()
