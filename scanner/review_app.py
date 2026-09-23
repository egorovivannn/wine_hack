"""Read-only browser for manually reviewed organizer photos and visual Top-20.

Start with ``uv run --locked uvicorn scanner.review_app:app --host <tailscale-ip> --port 18766``.
The UI never runs inference; it reads frozen review artifacts and whitelisted media.
"""

from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
import json
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from .catalog import ROOT, sha256_file
from .labels import read_labels

STATIC = Path(__file__).parent / "static"


class ReviewData:
    def __init__(
        self,
        labels_path: Path = ROOT / "evaluation/all_official_labels.tsv",
        photos_dir: Path = ROOT / "data/official_real_photos",
        visual_path: Path = ROOT / "data/evaluation/manual_review/visual_top20_all.json",
        ocr_path: Path = ROOT / "data/evaluation/manual_review/ocr_all.json",
        manifest_path: Path = ROOT / "data/catalog_manifest.json",
        references_dir: Path = ROOT / "data/competition_imgs",
        cache_dir: Path = ROOT / "data/evaluation/review_thumbs",
    ) -> None:
        self.labels = read_labels(labels_path)
        self.photos_dir = photos_dir
        self.references_dir = references_dir
        self.cache_dir = cache_dir
        self.visual = json.loads(visual_path.read_text(encoding="utf-8"))
        self.ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cards = {card["slug"]: card for card in manifest["cards"]}
        self.reference_hashes = {item["filename"]: item["sha256"] for item in manifest["images"]}
        self.allowed_references: set[str] = set()
        self._thumb_lock = Lock()

        actual_photos = {path.name for path in photos_dir.glob("*.webp")}
        expected = set(self.labels)
        if len(expected) != 100 or expected != actual_photos:
            raise ValueError("Review labels must cover exactly the 100 organizer photos")
        if expected != set(self.visual) or expected != set(self.ocr):
            raise ValueError("Visual/OCR reports must cover the same 100 photos")
        for name in expected:
            label = self.labels[name]
            if sha256_file(photos_dir / name) != label.image_sha256:
                raise ValueError(f"Organizer photo changed: {name}")
            if self.visual[name]["sha256"] != label.image_sha256:
                raise ValueError(f"Visual report belongs to another photo: {name}")
            if self.ocr[name]["sha256"] != label.image_sha256:
                raise ValueError(f"OCR report belongs to another photo: {name}")
            if label.slug and label.slug not in self.cards:
                raise ValueError(f"Unknown verified slug: {label.slug}")
            candidates = self.visual[name]["candidates"]
            if len(candidates) != 20:
                raise ValueError(f"Expected 20 visual candidates for {name}")
            for rank, candidate in enumerate(candidates, 1):
                slug, filename = candidate["slug"], candidate["filename"]
                if candidate["rank"] != rank or slug not in self.cards:
                    raise ValueError(f"Invalid candidate rank/slug for {name}")
                if self.cards[slug]["image_name"] != filename:
                    raise ValueError(f"Candidate-reference mismatch for {name}: {slug}")
                if filename not in self.reference_hashes or not (references_dir / filename).is_file():
                    raise ValueError(f"Missing candidate reference for {name}: {filename}")
                self.allowed_references.add(filename)
            if label.slug:
                verified_ref = self.cards[label.slug]["image_name"]
                if verified_ref in self.reference_hashes and (references_dir / verified_ref).is_file():
                    self.allowed_references.add(verified_ref)

        self.provenance = {
            "labels_sha256": sha256_file(labels_path),
            "visual_top20_sha256": sha256_file(visual_path),
            "ocr_sha256": sha256_file(ocr_path),
            "catalog_manifest_sha256": sha256_file(manifest_path),
        }
        self.counts = dict(Counter(label.status for label in self.labels.values()))

    def summary(self, name: str) -> dict[str, Any]:
        label = self.labels[name]
        candidates = self.visual[name]["candidates"]
        match_rank = next((c["rank"] for c in candidates if c["slug"] == label.slug), None) if label.slug else None
        return {
            "name": name,
            "status": label.status,
            "slug": label.slug or None,
            "evidence": label.evidence,
            "top1": {"slug": candidates[0]["slug"], "score": candidates[0]["score"],
                     "name": candidates[0]["name"], "winery": candidates[0]["winery"]},
            "verified_rank": match_rank,
            "photo_url": f"/media/photos/{name}",
            "thumb_url": f"/media/thumbs/{name}",
        }

    def detail(self, name: str) -> dict[str, Any]:
        if name not in self.labels:
            raise KeyError(name)
        result = self.summary(name)
        label = self.labels[name]
        result["sha256"] = label.image_sha256
        result["ocr_lines"] = self.ocr[name]["lines"]
        result["candidates"] = []
        for item in self.visual[name]["candidates"]:
            card = self.cards[item["slug"]]
            result["candidates"].append({
                **item,
                "card": {key: card[key] for key in ("name", "winery", "category", "color", "region", "grape", "description")},
                "reference_url": f"/media/references/{item['filename']}",
                "reference_sha256": self.reference_hashes[item["filename"]],
            })
        if label.slug:
            card = self.cards[label.slug]
            result["verified_card"] = card
            filename = card["image_name"]
            result["verified_reference_url"] = (f"/media/references/{filename}"
                                                 if filename in self.allowed_references else None)
        else:
            result["verified_card"] = None
            result["verified_reference_url"] = None
        return result

    def thumbnail(self, name: str) -> Path:
        if name not in self.labels:
            raise KeyError(name)
        output = self.cache_dir / f"{self.labels[name].image_sha256[:20]}.jpg"
        if output.is_file():
            return output
        with self._thumb_lock:
            if output.is_file():
                return output
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(self.photos_dir / name) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((360, 360), Image.Resampling.LANCZOS)
                temporary = output.with_suffix(".tmp")
                image.save(temporary, format="JPEG", quality=82, optimize=True)
                temporary.replace(output)
        return output


def create_app(data: ReviewData | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.review = data if data is not None else ReviewData()
        yield

    app = FastAPI(title="Wine photo review", version="1.0", lifespan=lifespan)

    def review() -> ReviewData:
        return app.state.review

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ready", "photos": len(review().labels)}

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        return {"total": len(review().labels), "counts": review().counts, "provenance": review().provenance}

    @app.get("/api/photos")
    def photos(status: str = Query("all"), q: str = Query("", max_length=200)) -> dict[str, Any]:
        if status not in {"all", "verified", "unknown", "ambiguous"}:
            raise HTTPException(status_code=400, detail="Invalid status")
        query = q.strip().casefold()
        rows = []
        for name in sorted(review().labels):
            item = review().summary(name)
            if status != "all" and item["status"] != status:
                continue
            searchable = " ".join((name, item["evidence"], item["slug"] or "",
                                   item["top1"]["name"], item["top1"]["winery"]))
            if query and query not in searchable.casefold():
                continue
            rows.append(item)
        return {"total": len(rows), "photos": rows}

    @app.get("/api/photos/{name}")
    def photo(name: str) -> dict[str, Any]:
        try:
            return review().detail(name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Photo not found") from error

    @app.get("/media/photos/{name}")
    def photo_media(name: str) -> FileResponse:
        if name not in review().labels:
            raise HTTPException(status_code=404, detail="Photo not found")
        return FileResponse(review().photos_dir / name, media_type="image/webp")

    @app.get("/media/thumbs/{name}")
    def photo_thumb(name: str) -> FileResponse:
        try:
            return FileResponse(review().thumbnail(name), media_type="image/jpeg")
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Photo not found") from error

    @app.get("/media/references/{name}")
    def reference_media(name: str) -> FileResponse:
        if name not in review().allowed_references:
            raise HTTPException(status_code=404, detail="Reference not found")
        return FileResponse(review().references_dir / name, media_type="image/webp")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "review.html", media_type="text/html")

    @app.get("/review.css")
    def css() -> FileResponse:
        return FileResponse(STATIC / "review.css", media_type="text/css")

    @app.get("/review.js")
    def js() -> FileResponse:
        return FileResponse(STATIC / "review.js", media_type="application/javascript")

    return app


app = create_app()
