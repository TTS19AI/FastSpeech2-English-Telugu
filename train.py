
"""Run the reference FastSpeech 2 trainer with English project configs."""
from pathlib import Path
import argparse
import os
import random
import runpy
import sys

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--stop_step", type=int, default=None,
        help="Absolute final step; defaults to config/train.yaml total_step."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    repository = Path("/content/FastSpeech2_reference")
    config_dir = project_dir / "config"

    paths = {
        name: config_dir / name
        for name in ("preprocess.yaml", "model.yaml", "train.yaml")
    }
    for path in [repository / "train.py", *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_config = yaml.safe_load(
        paths["train.yaml"].read_text(encoding="utf-8")
    )
    stop_step = (
        args.stop_step if args.stop_step is not None
        else train_config["step"]["total_step"]
    )
    if not 0 <= args.restore_step < stop_step:
        raise ValueError("Require 0 <= restore_step < stop_step.")

    checkpoint_dir = Path(train_config["path"]["ckpt_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    existing_steps = sorted(
        int(path.name.split(".")[0])
        for path in checkpoint_dir.glob("*.pth.tar")
        if path.name.split(".")[0].isdigit()
    )
    if existing_steps and max(existing_steps) > args.restore_step:
        raise RuntimeError(
            "A newer production checkpoint already exists. "
            "Choose its step for resumption before continuing."
        )
    if args.restore_step:
        checkpoint = checkpoint_dir / f"{args.restore_step}.pth.tar"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    train_config["step"]["total_step"] = stop_step
    run_dir = config_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_dir / (
        f"train_from_{args.restore_step}_to_{stop_step}.yaml"
    )
    run_config.write_text(
        yaml.safe_dump(train_config, sort_keys=False),
        encoding="utf-8",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable.")
    torch.cuda.manual_seed_all(args.seed)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Restore step:", args.restore_step, flush=True)
    print("Final step:", stop_step, flush=True)
    print("Checkpoint directory:", checkpoint_dir, flush=True)
    print("Run configuration:", run_config, flush=True)

    # Preserve the reference architecture and training implementation.
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    sys.argv = [
        str(repository / "train.py"),
        "--restore_step", str(args.restore_step),
        "-p", str(paths["preprocess.yaml"]),
        "-m", str(paths["model.yaml"]),
        "-t", str(run_config),
    ]
    runpy.run_path(str(repository / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
