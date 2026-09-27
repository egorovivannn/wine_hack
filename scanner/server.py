"""Local API for the organizer's evaluator and the web wine card."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import numpy as np
import torch
from PIL import UnidentifiedImageError

from .catalog import ROOT, sha256_file
from .image import decode_image, query_views
from .ocr import (MODEL_DIR as DEFAULT_OCR_MODEL_DIR, OCRReader, OCRReranker,
                  OCR_WEIGHT, VISUAL_MARGIN_FOR_OCR)
from .pairing import recommend_pairings
from .verifier import (CANDIDATES, DEFAULT_MODEL_DIR as DEFAULT_VERIFIER_DIR,
                       DEFAULT_THUMBNAIL_DIR, reference_thumbnail, sweetness)
from .vision import (DEFAULT_INDEX, DEFAULT_MANIFEST, DEFAULT_MODEL_DIR,
                     VisionEncoder, load_index)


MAX_UPLOAD_BYTES = 16 * 1024 * 1024
# Softmax temperature over the top-5 final scores. The result is a relative confidence
# among the shown candidates, not a calibrated probability of being in the catalog.
CONFIDENCE_TEMPERATURE = 0.02
# Below this the UI suggests checking the alternatives. It never changes the evaluator answer.
CONFIDENT_THRESHOLD = 0.7
STATIC_PAGE = Path(__file__).parent / "static/index.html"


class SearchEngine:
    def __init__(self, manifest_path: Path, index_path: Path, model_dir: Path,
                 images_dir: Path, use_ocr: bool = False,
                 ocr_model_dir: Path = DEFAULT_OCR_MODEL_DIR,
                 use_verifier: bool = False, verifier_dir: Path = DEFAULT_VERIFIER_DIR,
                 device: str | None = None, verifier_candidates: int = CANDIDATES,
                 verifier_center_crop: bool = False):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cards = {card["slug"]: card for card in manifest["cards"]}
        for card in self.cards.values():
            # Near-duplicate cards often differ only in sweetness; show it to the user too.
            found = sweetness(card)
            card["sweetness"] = None if found == "не указана" else found
        self.images_dir = images_dir
        self.gallery = load_index(index_path, manifest_path)
        index_metadata = json.loads(index_path.with_suffix(".json").read_text())
        if index_metadata["model_weights_sha256"] != sha256_file(model_dir / "model.safetensors"):
            raise ValueError("Model weights differ from the checkpoint used for the index")
        self.encoder = VisionEncoder(model_dir, device)
        self.ocr_model_dir = Path(ocr_model_dir)
        self.ocr_reader = OCRReader(self.ocr_model_dir) if use_ocr else None
        self.ocr_reranker = OCRReranker(self.cards) if use_ocr else None
        self.verifier = None
        self.candidates = verifier_candidates if use_verifier else CANDIDATES
        if use_verifier:
            from .verifier import CandidateVerifier
            self.verifier = CandidateVerifier(self.cards, images_dir, verifier_dir,
                                              device=device, candidates=verifier_candidates,
                                              center_crop=verifier_center_crop)
        self.lock = Lock()

    @classmethod
    def from_environment(cls) -> "SearchEngine":
        device = os.getenv("WINE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        # On CPU the verifier takes about a minute per photo, beyond the evaluator's
        # 10-second timeout, so it is enabled by default only on a GPU.
        verifier_default = "1" if device == "cuda" else "0"
        return cls(
            manifest_path=Path(os.getenv("WINE_MANIFEST", str(DEFAULT_MANIFEST))),
            index_path=Path(os.getenv("WINE_INDEX", str(DEFAULT_INDEX))),
            model_dir=Path(os.getenv("WINE_MODEL_DIR", str(DEFAULT_MODEL_DIR))),
            images_dir=Path(os.getenv("WINE_IMAGES_DIR", str(ROOT / "data/competition_imgs"))),
            use_ocr=os.getenv("WINE_OCR", "0") == "1",
            ocr_model_dir=Path(os.getenv("WINE_OCR_MODEL_DIR", str(DEFAULT_OCR_MODEL_DIR))),
            use_verifier=os.getenv("WINE_VERIFIER", verifier_default) == "1",
            verifier_dir=Path(os.getenv("WINE_VERIFIER_DIR", str(DEFAULT_VERIFIER_DIR))),
            device=device,
        )

    def get_card(self, slug: str) -> dict | None:
        return self.cards.get(slug)

    def get_image_path(self, slug: str) -> Path | None:
        card = self.get_card(slug)
        if card is None or not card["reference_available"]:
            return None
        path = self.images_dir / card["image_name"]
        return path if path.is_file() else None

    def get_thumbnail_path(self, slug: str) -> Path | None:
        """512 px copy of the reference for phones; catalog originals reach several MB."""
        path = self.get_image_path(slug)
        if path is None:
            return None
        reference_thumbnail(self.images_dir, path.name)
        return DEFAULT_THUMBNAIL_DIR / f"{path.name}.png"

    def warm_up(self) -> None:
        """Run one catalog image through the pipeline so the first real request is not slow."""
        card = next((card for card in self.cards.values()
                     if card["reference_available"] and self.get_image_path(card["slug"])), None)
        if card is not None:
            self.predict(self.get_image_path(card["slug"]).read_bytes())

    def _reference(self, candidate):
        return reference_thumbnail(self.images_dir, candidate.image_name)

    def predict(self, content: bytes) -> dict:
        image = decode_image(content)
        views = list(query_views(image))
        ocr_used = verifier_used = False
        with self.lock:
            vectors = self.encoder.encode(views)
            ranked = self.gallery.search(vectors, top_k=20)
            final = np.array([item.score for item in ranked[:self.candidates]])
            if self.verifier is not None:
                ranked, final = self.verifier.rerank(image, ranked, self._reference)
                verifier_used = True
            elif self.ocr_reader is not None:
                ocr_used = len(ranked) > 1 and (
                    ranked[0].score - ranked[1].score < VISUAL_MARGIN_FOR_OCR)
                lines = self.ocr_reader.read(image) if ocr_used else []
        if ocr_used:
            ranked = self.ocr_reranker.rerank(ranked, lines)
            final = np.array([item.score + OCR_WEIGHT * self.ocr_reranker.lexical_score(lines, item.slug)
                              for item in ranked[:self.candidates]])
        ranked = ranked[:6]
        first = ranked[0]
        top = ranked[:len(final)]
        probabilities = np.exp((final - final.max()) / CONFIDENCE_TEMPERATURE)
        probabilities /= probabilities.sum()
        confidence = float(probabilities[0])
        return {
            "slug": first.slug,
            "card": self.cards[first.slug],
            "confidence": confidence,
            "verdict": "confident" if confidence >= CONFIDENT_THRESHOLD else "check_alternatives",
            "top5": [{"slug": item.slug, "probability": float(probability),
                      "score": float(score)}
                     for item, probability, score in zip(top[:5], probabilities, final)],
            "visual_similarity": first.score,
            "margin_to_second": float(final[0] - final[1]) if len(final) > 1 else None,
            "ocr_used": ocr_used,
            "verifier_used": verifier_used,
            "shared_reference": bool(len(ranked) > 1 and
                                     ranked[1].image_name == first.image_name),
            "alternatives": [asdict(item) for item in ranked[1:]],
        }


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or SearchEngine.from_environment()
        if engine is None:
            app.state.engine.warm_up()
        yield

    app = FastAPI(title="Wine scanner", version="0.1.0", lifespan=lifespan)

    async def predict_upload(image: UploadFile) -> dict:
        content = await image.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Empty image")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 16 MiB")
        try:
            return await run_in_threadpool(app.state.engine.predict, content)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail="Image could not be decoded") from error

    @app.get("/health")
    def health():
        return {"status": "ready", "cards": len(app.state.engine.cards)}

    @app.post("/v1/eval/predict")
    async def eval_predict(image: UploadFile = File(...)):
        result = await predict_upload(image)
        return {"slug": result["slug"]}

    @app.post("/v1/predict")
    async def product_predict(image: UploadFile = File(...)):
        return await predict_upload(image)

    @app.get("/v1/wines/{slug}")
    def get_wine(slug: str):
        card = app.state.engine.get_card(slug)
        if card is None:
            raise HTTPException(status_code=404, detail="Unknown slug")
        return card

    @app.get("/v1/wines/{slug}/pairings")
    def get_pairings(slug: str, dish: str | None = None):
        card = app.state.engine.get_card(slug)
        if card is None:
            raise HTTPException(status_code=404, detail="Unknown slug")
        try:
            return recommend_pairings(card, dish)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/v1/wines/{slug}/image")
    def get_wine_image(slug: str):
        path = app.state.engine.get_image_path(slug)
        if path is None:
            raise HTTPException(status_code=404, detail="Image unavailable")
        return FileResponse(path)

    @app.get("/v1/wines/{slug}/thumbnail")
    def get_wine_thumbnail(slug: str):
        path = app.state.engine.get_thumbnail_path(slug)
        if path is None:
            raise HTTPException(status_code=404, detail="Image unavailable")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_PAGE)

    return app


app = create_app()
