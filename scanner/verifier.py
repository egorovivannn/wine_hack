"""Vision-language check that picks the exact card among close visual candidates.

Near-duplicate cards (same label, different sweetness, grape or vintage) are
indistinguishable for a global image embedding. The verifier shows the photo and the
reference images of the top candidates with their catalog text to Qwen3.5-4B and reads
the probability of each candidate number. That probability is fused with the visual score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .catalog import ROOT, sha256_file
from .vision import Candidate


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_MODEL_DIR = ROOT / "data/models/qwen3.5-4b"
MODEL_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors":
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors":
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
CANDIDATES = 5
# Final score = visual cosine + VERIFIER_WEIGHT * log P(candidate). Chosen on the official
# photos (plateau 0.015-0.03), so it is a development setting, not a held-out estimate.
VERIFIER_WEIGHT = 0.02
QUERY_SIDE = 1024
REFERENCE_SIDE = 512
SWEETNESS = (
    ("полусух", "полусухое"), ("polusuh", "полусухое"), ("p_suh", "полусухое"),
    ("полуслад", "полусладкое"), ("polusl", "полусладкое"), ("p_sl", "полусладкое"),
    ("экстра брют", "экстра брют"), ("ekstra-bryut", "экстра брют"),
    ("ekstra_bryut", "экстра брют"), ("брют", "брют"), ("bryut", "брют"),
    ("сладк", "сладкое"), ("сух", "сухое"), ("suh", "сухое"),
)


def sweetness(card: dict) -> str:
    """Sweetness is rarely a separate field; take it from name, slug, file name or text."""
    text = " ".join([card["name"], card["slug"], card["image_name"],
                     card.get("description", "")[:400]]).lower()
    return next((value for key, value in SWEETNESS if key in text), "не указана")


def describe(card: dict) -> str:
    return (f"{card['name'].strip()} | винодельня: {card['winery']} | {card['category']} | "
            f"сорт: {card['grape']} | сладость: {sweetness(card)}")


DEFAULT_THUMBNAIL_DIR = ROOT / "data/index/verifier_thumbnails"


def _thumbnail(image: Image.Image, side: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((side, side))
    return image


def reference_thumbnail(images_dir: Path, filename: str,
                        cache_dir: Path = DEFAULT_THUMBNAIL_DIR) -> Image.Image:
    """Catalog photos reach 5000 px; decoding five per request costs over a second."""
    from .image import decode_image

    cached = Path(cache_dir) / f"{filename}.png"
    if cached.is_file():
        return Image.open(cached).convert("RGB")
    image = _thumbnail(decode_image(Path(images_dir) / filename), REFERENCE_SIDE)
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".tmp.png")
    image.save(temporary)
    temporary.replace(cached)
    return image


class CandidateVerifier:
    def __init__(self, cards: dict[str, dict], images_dir: Path,
                 model_dir: Path = DEFAULT_MODEL_DIR, verify_hashes: bool = True,
                 device: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_dir = Path(model_dir)
        if verify_hashes:
            self.verify_weights(model_dir)
        self.cards = cards
        self.images_dir = Path(images_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            # 4-bit weights (about 3.3 GiB) leave room for SigLIP on a 10 GiB card.
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
                quantization_config=quantization).eval()
        else:
            # bitsandbytes 4-bit is GPU-only; float32 is the fastest portable CPU path.
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=torch.float32, local_files_only=True).eval()
        tokenizer = self.processor.tokenizer
        self.prefix = tokenizer("Кандидат ", add_special_tokens=False,
                                return_tensors="pt")["input_ids"]
        self.digit_ids = [tokenizer.convert_tokens_to_ids(str(number))
                          for number in range(1, CANDIDATES + 1)]

    @staticmethod
    def verify_weights(model_dir: Path) -> None:
        for filename, expected in MODEL_SHA256.items():
            path = Path(model_dir) / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Verifier weight missing or changed: {path}")

    def _messages(self, image: Image.Image, candidates: list[Candidate],
                  references: list[Image.Image]) -> list[dict]:
        content = [{"type": "text", "text": "Фото покупателя (целевая бутылка — в центре кадра):"},
                   {"type": "image", "image": _thumbnail(image, QUERY_SIDE)}]
        for number, (candidate, reference) in enumerate(zip(candidates, references), 1):
            content += [{"type": "text",
                         "text": f"Кандидат {number}: {describe(self.cards[candidate.slug])}"},
                        {"type": "image", "image": _thumbnail(reference, REFERENCE_SIDE)}]
        content.append({"type": "text", "text": (
            f"Какой кандидат (1–{len(candidates)}) — это ровно то же вино, что на фото? "
            "Сравни надписи на этикетке: название, сорт, сладость (сухое/полусухое/"
            "полусладкое/сладкое/брют), цвет и год, а также дизайн этикетки. "
            "Ответь только номером.")})
        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def log_probabilities(self, image: Image.Image, candidates: list[Candidate],
                          references: list[Image.Image]) -> np.ndarray:
        inputs = self.processor.apply_chat_template(
            self._messages(image, candidates, references), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False).to(self.model.device)
        # Force the answer prefix, then read the next-token distribution over digits.
        prefix = self.prefix.to(self.model.device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(prefix)], dim=1)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(prefix)], dim=1)
        logits = self.model(**inputs).logits[0, -1].float()
        ids = self.digit_ids[:len(candidates)]
        return torch.log_softmax(logits[ids], dim=0).cpu().numpy()

    def rerank(self, image: Image.Image, ranked: list[Candidate],
               load_reference) -> tuple[list[Candidate], np.ndarray]:
        """Return candidates sorted by fused score and the fused scores of the first K."""
        head = ranked[:CANDIDATES]
        if len(head) < 2:
            return ranked, np.array([candidate.score for candidate in head])
        log_probs = self.log_probabilities(
            image, head, [load_reference(candidate) for candidate in head])
        fused = np.array([candidate.score for candidate in head]) + VERIFIER_WEIGHT * log_probs
        order = np.argsort(-fused, kind="stable")
        return [head[i] for i in order] + ranked[CANDIDATES:], fused[order]


def download(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir,
                      allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"])
    CandidateVerifier.verify_weights(model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify", "thumbnails"])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/catalog_manifest_corrected.json")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/competition_imgs")
    args = parser.parse_args()
    if args.command == "download":
        download(args.model_dir)
    elif args.command == "thumbnails":
        import json

        images = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
        for record in images:
            reference_thumbnail(args.images_dir, record["filename"])
        print(f"{len(images)} verifier thumbnails ready in {DEFAULT_THUMBNAIL_DIR}")
        return
    else:
        CandidateVerifier.verify_weights(args.model_dir)
    print("Verifier checkpoint hashes verified")


if __name__ == "__main__":
    main()
