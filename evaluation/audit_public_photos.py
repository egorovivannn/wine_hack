"""Fetch a bounded public-photo pilot with provenance and duplicate evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import cv2
import httpx
import numpy as np
from PIL import Image, ImageOps

from scanner.catalog import ROOT, sha256_file

USER_AGENT = "WineScannerResearch/1.0 (public image provenance pilot)"
SKIP_HOSTS = {"t.me", "rvwa.ru", "barcode-list.ru"}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self.meta: dict[str, str] = {}
        self.images: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = values.get("property") or values.get("name")
            if key and values.get("content"):
                self.meta[key.lower()] = values["content"]
        if tag == "img":
            source = values.get("src") or values.get("data-src")
            if source:
                self.images.append((source, values.get("alt") or ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def phash(image: Image.Image) -> int:
    gray = np.asarray(ImageOps.grayscale(image).resize((32, 32)), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8].flatten()[1:]
    median = float(np.median(low))
    return sum(int(value > median) << index for index, value in enumerate(low))


def image_evidence(raw: bytes, reference_hash: int, reference_sha: str) -> dict:
    image = Image.open(io.BytesIO(raw))
    image.load()
    width, height = image.size
    return {
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "image_width": width,
        "image_height": height,
        "phash_hamming": (phash(image) ^ reference_hash).bit_count(),
        "exact_duplicate": hashlib.sha256(raw).hexdigest() == reference_sha,
    }


class RespectfulClient:
    def __init__(self, delay: float) -> None:
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=12
        )
        self.delay = delay
        self.last_request: dict[str, float] = {}
        self.robots: dict[str, RobotFileParser | None] = {}

    def close(self) -> None:
        self.client.close()

    def get(self, url: str) -> httpx.Response:
        host = urlsplit(url).netloc
        elapsed = time.monotonic() - self.last_request.get(host, 0)
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request[host] = time.monotonic()
        return self.client.get(url)

    def allowed(self, url: str) -> bool:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.hostname in SKIP_HOSTS
        ):
            return False
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            try:
                response = self.get(origin + "/robots.txt")
                if response.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
                    self.robots[origin] = parser
                else:
                    self.robots[origin] = None
            except httpx.HTTPError:
                self.robots[origin] = None
        parser = self.robots[origin]
        return parser is not None and parser.can_fetch(USER_AGENT, url)


def audit(
    selection: Path,
    search: Path,
    references: Path,
    output: Path,
    images_dir: Path,
    pages_per_slug: int,
    delay: float,
) -> None:
    with selection.open(encoding="utf-8", newline="") as stream:
        selected = {row["slug"]: row for row in csv.DictReader(stream, delimiter="\t")}
    searches = json.loads(search.read_text(encoding="utf-8"))["results"]
    if set(selected) != {row["slug"] for row in searches}:
        raise ValueError("Pilot selection and search results differ")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    client = RespectfulClient(delay)
    try:
        with output.open("w", encoding="utf-8") as stream:
            for position, item in enumerate(searches, 1):
                card = selected[item["slug"]]
                reference = references / card["reference_filename"]
                if sha256_file(reference) != card["reference_sha256"]:
                    raise ValueError(f"Reference changed for {item['slug']}")
                with Image.open(reference) as image:
                    reference_hash = phash(image)
                seen_hosts: set[str] = set()
                attempts = 0
                for hit in item["hits"]:
                    if attempts >= pages_per_slug:
                        break
                    url = hit["url"]
                    host = urlsplit(url).hostname or ""
                    if host in seen_hosts or urlsplit(url).path.lower().endswith(
                        ".pdf"
                    ):
                        continue
                    seen_hosts.add(host)
                    attempts += 1
                    row = {
                        "slug": item["slug"],
                        "page_url": url,
                        "search_title": hit["title"],
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "reference_sha256": card["reference_sha256"],
                        "status": "unreviewed",
                    }
                    try:
                        if not client.allowed(url):
                            row["status"] = "robots_or_scheme_reject"
                        else:
                            response = client.get(url)
                            row["page_status"] = response.status_code
                            row["page_sha256"] = hashlib.sha256(
                                response.content
                            ).hexdigest()
                            if (
                                response.status_code != 200
                                or "html"
                                not in response.headers.get("content-type", "")
                            ):
                                row["status"] = "page_unavailable"
                            else:
                                parser = PageParser()
                                parser.feed(response.text)
                                row["page_title"] = parser.title.strip()[:250]
                                row["page_description"] = parser.meta.get(
                                    "description", ""
                                )[:500]
                                image_url = parser.meta.get(
                                    "og:image"
                                ) or parser.meta.get("twitter:image")
                                if not image_url:
                                    image_url = next(
                                        (
                                            src
                                            for src, alt in parser.images
                                            if card["name"].casefold() in alt.casefold()
                                        ),
                                        None,
                                    )
                                if not image_url:
                                    row["status"] = "no_image_url"
                                else:
                                    image_url = urljoin(str(response.url), image_url)
                                    row["image_url"] = image_url
                                    if not client.allowed(image_url):
                                        row["status"] = "image_robots_reject"
                                    else:
                                        image_response = client.get(image_url)
                                        row["image_status"] = image_response.status_code
                                        if (
                                            image_response.status_code != 200
                                            or len(image_response.content) > 10_000_000
                                        ):
                                            row["status"] = "image_unavailable"
                                        else:
                                            row.update(
                                                image_evidence(
                                                    image_response.content,
                                                    reference_hash,
                                                    card["reference_sha256"],
                                                )
                                            )
                                            if (
                                                min(
                                                    row["image_width"],
                                                    row["image_height"],
                                                )
                                                < 150
                                            ):
                                                row["status"] = "image_too_small"
                                            else:
                                                filename = (
                                                    f"{item['slug']}_{attempts}.jpg"
                                                )
                                                with Image.open(
                                                    io.BytesIO(image_response.content)
                                                ) as image:
                                                    image.convert("RGB").save(
                                                        images_dir / filename,
                                                        quality=93,
                                                    )
                                                row["local_image"] = str(
                                                    images_dir / filename
                                                )
                                                row["status"] = (
                                                    "duplicate_candidate"
                                                    if row["exact_duplicate"]
                                                    or row["phash_hamming"] <= 8
                                                    else "needs_review"
                                                )
                    except (httpx.HTTPError, OSError, ValueError) as error:
                        row["status"] = "fetch_error"
                        row["error"] = str(error)[:160]
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    stream.flush()
                if position % 10 == 0:
                    print(f"Audited {position}/{len(searches)} slugs", flush=True)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "evaluation/manifests/catalog_photo_pilot_100.tsv",
    )
    parser.add_argument(
        "--search",
        type=Path,
        default=ROOT / "data/external/catalog_photo_pilot_search.json",
    )
    parser.add_argument(
        "--references", type=Path, default=ROOT / "data/competition_imgs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/external/catalog_photo_pilot_audit.jsonl",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "data/external/catalog_photo_pilot_images",
    )
    parser.add_argument("--pages-per-slug", type=int, default=2)
    parser.add_argument("--host-delay", type=float, default=1.0)
    args = parser.parse_args()
    audit(
        args.selection,
        args.search,
        args.references,
        args.output,
        args.images_dir,
        args.pages_per_slug,
        args.host_delay,
    )


if __name__ == "__main__":
    main()
