"""Optional OCR evidence for close visual candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import torch
from unidecode import unidecode
from rapidfuzz.fuzz import ratio

from .catalog import ROOT, sha256_file
from .vision import Candidate


MODEL_DIR = ROOT / "data/models/easyocr"
MODEL_SHA256 = {
    "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    "cyrillic_g2.pth": "48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29",
}
VISUAL_MARGIN_FOR_OCR = 0.015
OCR_WEIGHT = 0.03
STOP_WORDS = {
    "vino", "vinodelnia", "semeinaia", "semeinaiavinodelnia", "sukhoe",
    "krasnoe", "beloe", "rozovoe", "rossiiskoe", "rossiia", "godurozhaia",
    "kuban", "anapa", "vynograd", "vynograda", "sukhiekrasnoe",
    "sukhoekrasnoe", "wine", "winery",
}
YEAR = re.compile(r"(?:19|20)[0-9]{2}")


def normalize(value: str) -> str:
    """Put Cyrillic and Latin label words in one comparison alphabet."""
    return "".join(re.findall(r"[a-z0-9]+", unidecode(value).lower())).replace("c", "k")


def _fragments(card: dict) -> set[str]:
    fragments: set[str] = set()
    for field in ("name", "winery", "grape"):
        words = [word.replace("c", "k") for word in
                 re.findall(r"[a-z0-9]+", unidecode(card[field]).lower())]
        fragments.update(word for word in words if len(word) >= 4)
        fragments.update(words[index] + words[index + 1] for index in range(len(words) - 1))
        fragments.add("".join(words))
    words = [word.replace("c", "k") for word in card["slug"].split("-")]
    fragments.update(word for word in words if len(word) >= 4)
    fragments.update(words[index] + words[index + 1] for index in range(len(words) - 1))
    return {part for part in fragments if len(part) >= 4 and
            (not part.isdigit() or YEAR.fullmatch(part))}


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float


class OCRReranker:
    def __init__(self, cards: dict[str, dict]):
        self.fragments = {slug: _fragments(card) for slug, card in cards.items()
                          if card["reference_available"]}
        self.frequencies: Counter[str] = Counter()
        for parts in self.fragments.values():
            self.frequencies.update(parts)
        self.catalog_size = len(self.fragments)

    def lexical_score(self, lines: list[OCRLine], slug: str) -> float:
        fragments = self.fragments[slug]
        hits = []
        for line in lines:
            token = normalize(line.text)
            if (line.confidence < 0.5 or len(token) < 4 or len(token) > 35 or
                    token in STOP_WORDS or token.isdigit() and not YEAR.fullmatch(token)):
                continue
            similarity, matched = max(((ratio(token, part), part) for part in fragments),
                                      default=(0.0, ""))
            threshold = 98 if len(token) <= 4 else 93 if len(token) <= 6 else 90
            if similarity >= threshold:
                idf = math.log((self.catalog_size + 1) / (self.frequencies[matched] + 1))
                hits.append(line.confidence * idf * similarity / 100)
        return sum(sorted(hits, reverse=True)[:3])

    def rerank(self, candidates: list[Candidate], lines: list[OCRLine]) -> list[Candidate]:
        """Use OCR only when visual ranking cannot clearly separate first two wines."""
        if len(candidates) < 2 or candidates[0].score - candidates[1].score >= VISUAL_MARGIN_FOR_OCR:
            return candidates
        return sorted(candidates,
                      key=lambda item: item.score + OCR_WEIGHT * self.lexical_score(lines, item.slug),
                      reverse=True)


class OCRReader:
    def __init__(self, model_dir: Path = MODEL_DIR, allow_download: bool = False):
        import easyocr

        model_dir = Path(model_dir)
        if not allow_download:
            self._verify_weights(model_dir)
        self.reader = easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available(),
                                     model_storage_directory=str(model_dir),
                                     download_enabled=allow_download, verbose=False)
        self._verify_weights(model_dir)

    @staticmethod
    def _verify_weights(model_dir: Path) -> None:
        for filename, expected in MODEL_SHA256.items():
            path = model_dir / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"OCR model weight missing or changed: {path}")

    def read(self, image) -> list[OCRLine]:
        image = image.copy()
        image.thumbnail((2000, 2000))
        found = self.reader.readtext(np.asarray(image), detail=1, paragraph=False)
        return [OCRLine(text, float(confidence)) for _, text, confidence in found]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify"])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    if args.command == "download":
        OCRReader(args.model_dir, allow_download=True)
    else:
        OCRReader._verify_weights(args.model_dir)
    print("OCR checkpoint hashes verified")


if __name__ == "__main__":
    main()
