"""Download the exact SigLIP 2 checkpoint used by the scanner."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .vision import DEFAULT_MODEL_DIR, MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
        local_dir=args.output,
    )
    print(f"Checkpoint {MODEL_ID}@{MODEL_REVISION} saved in {path}")


if __name__ == "__main__":
    main()
