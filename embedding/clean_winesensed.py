"""Audit WineSensed repeats and make a deterministic leakage-aware manifest.

Input is the verified JPEG export of prepare_winesensed_sample. The output is
small metadata; images remain in the ignored input directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scanner.image import decode_image


def perceptual_hash(path: Path) -> int:
    image = cv2.cvtColor(np.asarray(decode_image(path)), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    frequencies = cv2.dct(np.float32(small))[:8, :8].ravel()
    bits = frequencies > np.median(frequencies[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


class Groups:
    def __init__(self, names: list[str]):
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])
        return self.parent[name]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def near_pairs(values: np.ndarray, threshold: int = 5):
    """Yield close 64-bit hashes without creating an all-pairs matrix."""
    for start in range(0, len(values), 256):
        distances = np.bitwise_count(
            np.bitwise_xor(values[start : start + 256, None], values[None, :])
        ).astype(np.uint8)
        for offset, other in np.argwhere(distances <= threshold):
            first = start + int(offset)
            if first < other:
                yield first, int(other), int(distances[offset, other])


def split_for(group: str) -> str:
    value = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 2**32
    return "test" if value < 0.15 else "val" if value < 0.30 else "train"


def clean(
    frame: pd.DataFrame, hashes: list[int], prior_vintages: set[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy().sort_values(["vintage_id", "filename"]).reset_index(drop=True)
    if frame.filename.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError("Source contains duplicate names or bytes")
    groups = Groups(frame.vintage_id.unique().tolist())
    visual_groups = Groups(frame.vintage_id.unique().tolist())
    known = frame[frame.winery_id.notna() & frame.winery_id.ne("")]
    for _, members in known.groupby("winery_id"):
        ids = sorted(members.vintage_id.unique())
        for vintage in ids[1:]:
            groups.union(ids[0], vintage)
    values = np.asarray(hashes, dtype=np.uint64)
    cross_class = []
    within_class = []
    for a, b, distance in near_pairs(values):
        if frame.vintage_id.iloc[a] == frame.vintage_id.iloc[b]:
            within_class.append((a, b, distance))
        else:
            groups.union(frame.vintage_id.iloc[a], frame.vintage_id.iloc[b])
            visual_groups.union(frame.vintage_id.iloc[a], frame.vintage_id.iloc[b])
            cross_class.append((a, b, distance))
    # A near-identical second photograph does not establish a real positive pair.
    discarded = set()
    for a, b, _ in within_class:
        if a not in discarded:
            discarded.add(b)
    frame["phash"] = [f"{value:016x}" for value in hashes]
    frame["identity_group"] = frame.vintage_id.map(groups.find)
    frame["ambiguity_group"] = frame.vintage_id.map(visual_groups.find)
    prior_vintages = prior_vintages or set()
    forced_groups = {
        groups.find(vintage) for vintage in prior_vintages if vintage in groups.parent
    }
    frame["split"] = frame.identity_group.map(
        lambda group: "train" if group in forced_groups else split_for(group)
    )
    cross_class_ids = {frame.vintage_id.iloc[a] for a, _, _ in cross_class} | {
        frame.vintage_id.iloc[b] for _, b, _ in cross_class
    }
    frame = frame.drop(index=list(discarded)).reset_index(drop=True)
    counts = frame.vintage_id.value_counts()
    frame = frame[frame.vintage_id.isin(counts[counts >= 2].index)].copy()

    def portable_path(value: str) -> str:
        path = Path(value).resolve()
        try:
            return str(path.relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

    frame["path"] = frame.path.map(portable_path)
    report = dict(
        source_rows=len(hashes),
        retained_rows=len(frame),
        retained_classes=int(frame.vintage_id.nunique()),
        near_duplicate_pairs_same_class=len(within_class),
        near_duplicate_pairs_cross_class=len(cross_class),
        classes_in_cross_class_pairs=len(cross_class_ids),
        split_images=dict(Counter(frame.split)),
        split_classes={
            s: int(frame[frame.split == s].vintage_id.nunique())
            for s in ("train", "val", "test")
        },
        unknown_winery_classes=int(
            frame.groupby("vintage_id").winery_id.first().isin([""]).sum()
        ),
        capture_sessions="not published; independence cannot be proven",
    )
    report["previously_evaluated_vintages_forced_train"] = len(
        set(frame.vintage_id) & prior_vintages
    )
    if frame.groupby("vintage_id").split.nunique().max() > 1:
        raise ValueError("Vintage crosses split")
    if frame.groupby("identity_group").split.nunique().max() > 1:
        raise ValueError("Connected near-duplicate or winery group crosses split")
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        help="Previously evaluated vintages forced into train",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    source = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    for row in source.itertuples():
        if hashlib.sha256(Path(row.path).read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Image SHA differs: {row.filename}")
    hashes = [
        perceptual_hash(Path(path))
        for path in source.sort_values(["vintage_id", "filename"]).path
    ]
    prior = (
        set(pd.read_csv(args.prior_manifest, dtype=str).vintage_id)
        if args.prior_manifest
        else set()
    )
    frame, report = clean(source, hashes, prior)
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "manifest.csv", index=False)
    report.update(
        source_manifest_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        prior_manifest_sha256=hashlib.sha256(
            args.prior_manifest.read_bytes()
        ).hexdigest()
        if args.prior_manifest
        else None,
        manifest_sha256=hashlib.sha256(
            (args.output / "manifest.csv").read_bytes()
        ).hexdigest(),
        hash_method="32x32 grayscale DCT 8x8; Hamming <=5; same-class duplicates removed, cross-class pairs grouped",
    )
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
