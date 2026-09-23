"""Local API for the organizer's evaluator and the mobile wine card."""

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
from PIL import UnidentifiedImageError

from .catalog import ROOT, sha256_file
from .image import decode_image, query_views
from .vision import (DEFAULT_INDEX, DEFAULT_MANIFEST, DEFAULT_MODEL_DIR,
                     VisionEncoder, load_index)


MAX_UPLOAD_BYTES = 16 * 1024 * 1024
STATIC_PAGE = Path(__file__).parent / "static/index.html"


class SearchEngine:
    def __init__(self, manifest_path: Path, index_path: Path, model_dir: Path,
                 images_dir: Path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cards = {card["slug"]: card for card in manifest["cards"]}
        self.images_dir = images_dir
        self.gallery = load_index(index_path, manifest_path)
        index_metadata = json.loads(index_path.with_suffix(".json").read_text())
        if index_metadata["model_weights_sha256"] != sha256_file(model_dir / "model.safetensors"):
            raise ValueError("Model weights differ from the checkpoint used for the index")
        self.encoder = VisionEncoder(model_dir)
        self.lock = Lock()

    @classmethod
    def from_environment(cls) -> "SearchEngine":
        return cls(
            manifest_path=Path(os.getenv("WINE_MANIFEST", str(DEFAULT_MANIFEST))),
            index_path=Path(os.getenv("WINE_INDEX", str(DEFAULT_INDEX))),
            model_dir=Path(os.getenv("WINE_MODEL_DIR", str(DEFAULT_MODEL_DIR))),
            images_dir=Path(os.getenv("WINE_IMAGES_DIR", str(ROOT / "data/competition_imgs"))),
        )

    def get_card(self, slug: str) -> dict | None:
        return self.cards.get(slug)

    def get_image_path(self, slug: str) -> Path | None:
        card = self.get_card(slug)
        if card is None or not card["reference_available"]:
            return None
        path = self.images_dir / card["image_name"]
        return path if path.is_file() else None

    def predict(self, content: bytes) -> dict:
        image = decode_image(content)
        views = list(query_views(image))
        with self.lock:
            vectors = self.encoder.encode(views)
        ranked = self.gallery.search(vectors, top_k=6)
        first = ranked[0]
        margin = first.score - ranked[1].score if len(ranked) > 1 else None
        return {
            "slug": first.slug,
            "card": self.cards[first.slug],
            "visual_similarity": first.score,
            "margin_to_second": margin,
            "shared_reference": bool(len(ranked) > 1 and
                                     ranked[1].image_name == first.image_name),
            "alternatives": [asdict(item) for item in ranked[1:]],
        }


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or SearchEngine.from_environment()
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

    @app.get("/v1/wines/{slug}/image")
    def get_wine_image(slug: str):
        path = app.state.engine.get_image_path(slug)
        if path is None:
            raise HTTPException(status_code=404, detail="Image unavailable")
        return FileResponse(path)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_PAGE)

    return app


app = create_app()
