"""Fetch a small, identity-labeled WineSensed sample from the official Figshare ZIP."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import time
from zipfile import ZipFile

import pandas as pd
import requests
from PIL import Image


ARTICLE = "https://api.figshare.com/v2/articles/23376560"
METADATA_MD5 = "6faa2bd1432412c4b41347e5e1607fef"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RemoteZip(io.RawIOBase):
    """Seekable HTTP Range reader; ZIP CRC is still verified by zipfile on read."""

    def __init__(self, url: str):
        self.http = requests.Session()
        self.source_url = url
        response = self.http.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Figshare server did not honor HTTP Range")
        self.url = response.url
        self.signed_at = time.monotonic()
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        start = self.position
        for attempt in range(4):
            try:
                # Figshare's redirected S3 signature can expire after only ten seconds.
                url = self.url if time.monotonic() - self.signed_at < 7 else self.source_url
                response = self.http.get(url, headers={"Range": f"bytes={start}-{start+size-1}"}, timeout=90)
                if response.status_code == 403 and url != self.source_url:
                    response = self.http.get(self.source_url, headers={"Range": f"bytes={start}-{start+size-1}"}, timeout=90)
                if response.status_code == 206 and response.url != self.url:
                    self.url = response.url
                    self.signed_at = time.monotonic()
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != size:
                    raise IOError(f"Unexpected Range result: {response.status_code}, {len(response.content)}")
                self.position += size
                return response.content
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
        raise AssertionError("unreachable")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=Path("data/external/winesensed/metadata.zip"))
    p.add_argument("--output", type=Path, default=Path("data/external/winesensed/sample_v1"))
    p.add_argument("--chunk", default="chunk_001.zip")
    p.add_argument("--classes", type=int, default=300)
    p.add_argument("--images-per-class", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if hashlib.md5(args.metadata.read_bytes()).hexdigest() != METADATA_MD5:
        p.error("Official metadata.zip MD5 mismatch")
    article = requests.get(ARTICLE, timeout=30).json()
    source = next(f for f in article["files"] if f["name"] == args.chunk)
    remote = RemoteZip(source["download_url"])
    if remote.size != source["size"]:
        p.error("Remote ZIP size differs from Figshare manifest")
    with ZipFile(remote) as archive, ZipFile(args.metadata) as metadata:
        members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".jpg")}
        with metadata.open("images_reviews_attributes.csv") as f:
            frame = pd.read_csv(f, usecols=["vintage_id", "image", "winery_id"], dtype=str).dropna(subset=["vintage_id", "image"])
        frame["filename"] = frame.image.str.rsplit("/", n=1).str[-1]
        frame = frame[frame.filename.isin(members)].drop_duplicates(["filename", "vintage_id"])
        conflicts = set(frame.groupby("filename").vintage_id.nunique().loc[lambda v: v > 1].index)
        frame = frame[~frame.filename.isin(conflicts)].drop_duplicates("filename")
        groups = {str(k): sorted(g.filename.tolist()) for k, g in frame.groupby("vintage_id") if len(g) >= args.images_per_class}
        wineries = frame.groupby("vintage_id").winery_id.first().to_dict()
        wineries = {key: value for key, value in wineries.items() if pd.notna(value) and value}
        rng = random.Random(args.seed)
        keys = sorted(groups)
        rng.shuffle(keys)
        # Winery-disjoint where metadata supplies an ID; otherwise vintage-disjoint.
        winery_groups = defaultdict(list)
        for vintage in keys:
            winery_groups[str(wineries.get(vintage) or "vintage:" + vintage)].append(vintage)
        units = list(winery_groups.values())
        rng.shuffle(units)
        chosen = [v for unit in units for v in unit][:args.classes]
        report = dict(source_article=ARTICLE, chunk_file_id=source["id"],
                      chunk_declared_md5=source["computed_md5"], chunk_bytes=source["size"],
                      metadata_md5=METADATA_MD5, metadata_sha256=sha256(args.metadata),
                      available_images=len(members), mapped_images=len(frame),
                      conflicting_identities=len(conflicts), eligible_vintages=len(groups),
                      chosen_vintages=len(chosen), requested_images=len(chosen) * args.images_per_class,
                      seed=args.seed)
        print(json.dumps(report, indent=2), flush=True)
        if args.dry_run:
            return
        if args.output.exists():
            p.error("Output exists; choose a fresh directory")
        args.output.mkdir(parents=True)
        # Assign whole winery units to train/val/test. No identity crosses split.
        chosen_set = set(chosen)
        chosen_units = [[v for v in unit if v in chosen_set] for unit in units]
        chosen_units = [unit for unit in chosen_units if unit]
        targets = {"test": .15 * len(chosen), "val": .15 * len(chosen)}
        sizes = Counter()
        split_by_vintage = {}
        for unit in chosen_units:
            split = min(targets, key=lambda s: sizes[s] / targets[s]) if any(sizes[s] < targets[s] for s in targets) else "train"
            for vintage in unit:
                split_by_vintage[vintage] = split
            sizes[split] += len(unit)
        rows = []
        seen_hashes = {}
        for number, vintage in enumerate(chosen, 1):
            for filename in groups[vintage][:args.images_per_class]:
                content = archive.read(members[filename])
                with Image.open(io.BytesIO(content)) as im:
                    im.verify()
                digest = hashlib.sha256(content).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes[digest] = vintage
                dest = args.output / "images" / filename
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(content)
                rows.append(dict(path=str(dest.resolve()), filename=filename, vintage_id=vintage,
                                 winery_id=str(wineries.get(vintage) or ""),
                                 split=split_by_vintage[vintage], sha256=digest))
            if number % 25 == 0:
                print(f"Fetched {number}/{len(chosen)} vintages", flush=True)
        frame_out = pd.DataFrame(rows)
        frame_out.to_csv(args.output / "manifest.csv", index=False)
        report.update(downloaded_images=len(rows), downloaded_classes=int(frame_out.vintage_id.nunique()),
                      split_images=dict(Counter(frame_out.split)),
                      split_classes={s: int(frame_out[frame_out.split == s].vintage_id.nunique()) for s in ("train", "val", "test")},
                      manifest_sha256=sha256(args.output / "manifest.csv"))
        (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
