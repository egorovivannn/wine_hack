"""Score saved predictions against verified labels without rerunning GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import sha256_file
from .labels import read_labels


def score_predictions(predictions_path: Path, labels_path: Path) -> dict:
    labels = read_labels(labels_path)
    predictions = {}
    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row["image_path"]
            if name in predictions:
                raise ValueError(f"Duplicate prediction for {name}")
            if (not isinstance(row["predicted_slug"], str) or not row["predicted_slug"] or
                    not isinstance(row["top5_slugs"], list) or
                    not 1 <= len(row["top5_slugs"]) <= 5 or
                    row["top5_slugs"][0] != row["predicted_slug"] or
                    any(not isinstance(slug, str) or not slug for slug in row["top5_slugs"])):
                raise ValueError(f"Malformed prediction for {name}")
            predictions[name] = row
    if not predictions:
        raise ValueError("Prediction file is empty")
    missing = set(labels) - set(predictions)
    if missing:
        raise ValueError(f"Missing predictions for {sorted(missing)[:3]}")
    verified = []
    for name, label in labels.items():
        row = predictions[name]
        if row["image_sha256"] != label.image_sha256:
            raise ValueError(f"Photo hash differs from manual label: {name}")
        if label.status == "verified":
            verified.append((name, label.slug, row))
    top1 = sum(row["predicted_slug"] == slug for _, slug, row in verified)
    top5 = sum(slug in row["top5_slugs"] for _, slug, row in verified)
    return {
        "predictions_sha256": sha256_file(predictions_path),
        "labels_sha256": sha256_file(labels_path),
        "images_predicted": len(predictions),
        "verified_labels": len(verified),
        "unknown_labels": sum(label.status == "unknown" for label in labels.values()),
        "ambiguous_labels": sum(label.status == "ambiguous" for label in labels.values()),
        "top1_correct": top1,
        "top5_correct": top5,
        "top1_accuracy": top1 / len(verified) if verified else None,
        "top5_accuracy": top5 / len(verified) if verified else None,
        "top1_errors": [{"image_path": name, "verified_slug": slug,
                         "predicted_slug": row["predicted_slug"]}
                        for name, slug, row in verified if row["predicted_slug"] != slug],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_predictions(args.predictions, args.labels)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
