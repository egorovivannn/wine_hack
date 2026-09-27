"""Catalog-only recommendations: analogues from other wineries and a question-driven sommelier.

Everything is computed from card fields (category, name, grapes, region, winery,
description). No sales, price or rating data exist in the catalog, so the scores express
style similarity and food fit as described in the cards, not popularity or quality.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from .pairing import DISHES, recommend_pairings


COLORS = {"red": "Красное", "white": "Белое", "rose": "Розовое", "orange": "Оранжевое"}
SWEETNESS_LEVELS = ("сухое", "полусухое", "полусладкое", "сладкое")
SPARKLING = re.compile(r"брют|bryut|brut|игрист|igrist|pet-?nat|петнат|пет-нат|шампанск|frizzante|фризанте")
WORD = re.compile(r"[а-яё]{4,}")
# Shared adjectives ("белые", "тёмных") explain nothing to a reader; show noun-like notes only.
ADJECTIVE = re.compile(r"(ые|ых|ый|ий|ая|яя|ое|ее|ой|ую|юю|ым|им|ими|ыми|ого|его|ому|ему)$")
# Frequent card words that describe nothing about taste.
STOP = set("""вино вина вином винограда виноград винодельни винодельня аромат аромате ароматы ароматом
вкус вкусе вкуса вкусом послевкусие послевкусии нотами ноты нотки нотками тона тонами оттенки
оттенками оттенком цвет цвета цвете яркий яркие яркое хорошо очень также может более которые которое
который этого этот этом этой сорта сорт сортов года годом лет подходит подойдет подойдёт отлично
идеально сочетается сочетания сочетание блюдам блюдами блюда блюд выдержка выдержки выдержано
выдерживается бочках бочке дубовых дубе нержавеющей стали ферментация урожая урожай производства
собранного собранный ручной ручного свежий свежие свежее свежая""".split())
TYPICAL = {
    "fish": "белое сухое или игристое брют", "seafood": "белое сухое или игристое брют",
    "poultry": "белое или розовое", "red_meat": "сухое красное", "vegetables": "розовое или белое",
    "cheese": "красное, оранжевое или сладкое", "dessert": "сладкое или полусладкое", "spicy": "полусладкое белое или розовое",
}
COLOR_PHRASE = {"red": "среди красных", "white": "среди белых", "rose": "среди розовых",
                "orange": "среди оранжевых", "sparkling": "среди игристых"}
DISH_WORDS = {
    "fish": ("рыб",), "seafood": ("морепродукт", "устриц", "мидии", "креветк"),
    "poultry": ("птиц", "куриц", "утк", "индейк"), "red_meat": ("мяс", "стейк", "говяд", "баранин", "дичь", "шашлык"),
    "vegetables": ("овощ", "салат"), "cheese": ("сыр",), "dessert": ("десерт", "фрукт", "выпечк", "шоколад"),
    "spicy": ("остр", "азиатск", "пряны"),
}


def color_key(card: dict) -> str:
    category = card.get("category", "").lower()
    return next((key for key, label in COLORS.items() if label.lower() in category), "white")


def is_sparkling(card: dict) -> bool:
    title = " ".join([card.get("name", ""), card["slug"], card.get("image_name", "")]).lower()
    # Some sparkling cards say so only in the opening line of the description.
    opening = card.get("description", "")[:250].lower()
    return bool(SPARKLING.search(title) or re.search(r"игрист|\bбрют|шампанск|пет-?нат|петнат", opening))


def sweetness_level(card: dict) -> str | None:
    """Whole-word sweetness from the product name first, then the description."""
    title = " ".join([card.get("name", ""), card["slug"].replace("-", " "),
                      card.get("image_name", "").replace("_", " ")]).lower()
    for pattern, level in ((r"полусух|polusuh|\bp suh\b", "полусухое"), (r"полуслад|polusl|\bp sl\b", "полусладкое"),
                           (r"\bсладкое\b|\bsladkoe\b", "сладкое"), (r"\bсухое\b|\bsuhoe\b|\bsuh\b", "сухое")):
        if re.search(pattern, title):
            return level
    found = re.search(r"\b(полусухое|полусладкое|сухое|сладкое)\s+(белое|красное|розовое|вино)",
                      card.get("description", "").lower())
    return found.group(1) if found else None


def grapes(card: dict) -> set[str]:
    return {part.strip().lower() for part in card.get("grape", "").split(",") if part.strip()}


@dataclass(frozen=True)
class Profile:
    color: str
    sparkling: bool
    sweetness: str | None
    grapes: frozenset[str]
    region: str
    winery: str
    terms: dict[str, float]


class Recommender:
    def __init__(self, cards: dict[str, dict]):
        self.cards = {slug: card for slug, card in cards.items()}
        documents = {slug: Counter(word for word in WORD.findall(card.get("description", "").lower())
                                   if word not in STOP)
                     for slug, card in self.cards.items()}
        frequency = Counter(word for counts in documents.values() for word in counts)
        total = len(documents)
        self.profiles: dict[str, Profile] = {}
        for slug, card in self.cards.items():
            weights = {word: count * math.log((1 + total) / (1 + frequency[word]))
                       for word, count in documents[slug].items()
                       # Words in more than 15 % of cards are generic; single-card words are noise.
                       if 1 < frequency[word] < 0.15 * total}
            top = dict(sorted(weights.items(), key=lambda item: -item[1])[:25])
            norm = math.sqrt(sum(value * value for value in top.values())) or 1.0
            self.profiles[slug] = Profile(
                color=color_key(card), sparkling=is_sparkling(card), sweetness=sweetness_level(card),
                grapes=frozenset(grapes(card)), region=card.get("region", ""), winery=card.get("winery", ""),
                terms={word: value / norm for word, value in top.items()})

    def _text_similarity(self, first: Profile, second: Profile) -> tuple[float, list[str]]:
        shared = first.terms.keys() & second.terms.keys()
        score = sum(first.terms[word] * second.terms[word] for word in shared)
        notes = [word for word in sorted(shared, key=lambda word: -(first.terms[word] * second.terms[word]))
                 if not ADJECTIVE.search(word)][:3]
        return score, notes

    def analogues(self, slug: str, limit: int = 4) -> list[dict]:
        """Wines of the same colour and type from other wineries, most similar first."""
        if slug not in self.profiles:
            raise KeyError(slug)
        base = self.profiles[slug]
        scored = []
        for other, profile in self.profiles.items():
            if (other == slug or profile.winery == base.winery or profile.color != base.color or
                    profile.sparkling != base.sparkling or
                    self.cards[other].get("reference_available") in (False, "False")):
                continue
            union = base.grapes | profile.grapes
            grape_overlap = len(base.grapes & profile.grapes) / len(union) if union else 0.0
            same_sweetness = base.sweetness is not None and base.sweetness == profile.sweetness
            text, notes = self._text_similarity(base, profile)
            score = 3 * grape_overlap + 1.5 * same_sweetness + 0.7 * (profile.region == base.region) + 3 * text
            reasons = []
            common = sorted(base.grapes & profile.grapes)
            if common:
                reasons.append("тот же сорт: " + ", ".join(common[:2]))
            if same_sweetness:
                reasons.append(f"тоже {profile.sweetness}")
            if profile.region == base.region and profile.region:
                reasons.append(f"тоже {profile.region}")
            if notes and text > 0.05:
                reasons.append("похожие ноты: " + ", ".join(notes))
            scored.append((score, other, reasons))
        scored.sort(key=lambda item: (-item[0], item[1]))
        result, wineries = [], set()
        for score, other, reasons in scored:
            winery = self.profiles[other].winery
            if winery in wineries:
                continue
            wineries.add(winery)
            result.append({"slug": other, "card": self.cards[other], "score": round(score, 4),
                           "reasons": reasons or ["тот же стиль вина"]})
            if len(result) == limit:
                break
        return result

    def sommelier(self, dish: str | None = None, color: str | None = None,
                  sweetness: str | None = None, like: str | None = None,
                  limit: int = 3, offset: int = 0) -> dict:
        """Pick wines for the answers; each pick explains which answers it satisfies."""
        if dish is not None and dish not in DISHES:
            raise ValueError(f"Unknown dish: {dish}")
        if color is not None and color not in (*COLORS, "sparkling"):
            raise ValueError(f"Unknown colour: {color}")
        if sweetness is not None and sweetness not in ("dry", "semi", "sweet"):
            raise ValueError(f"Unknown sweetness: {sweetness}")
        if like is not None and like not in self.profiles:
            raise ValueError(f"Unknown wine: {like}")
        base = self.profiles[like] if like else None
        notes = []
        scored = self._score(dish, color, sweetness, like, base)
        if not scored and sweetness:
            # e.g. no sweet sparkling in the catalog: keep the colour, drop the sweetness filter.
            scored = self._score(dish, color, None, like, base)
            if scored:
                notes.append(f"Вин с такой сладостью {COLOR_PHRASE.get(color, 'в каталоге')} нет — "
                             "показываю ближайшие без учёта сладости.")
        if dish and scored and not any(strong for _, _, _, strong in scored):
            notes.append(f"Прямых рекомендаций к блюду «{DISHES[dish].lower()}» среди таких вин нет; "
                         f"к нему обычно берут {TYPICAL[dish]}. Вот ближайшие по стилю.")
        scored.sort(key=lambda item: (-item[0], item[1]))
        picks, wineries = [], set()
        for score, slug, reasons, _ in scored:
            winery = self.profiles[slug].winery
            if winery in wineries:
                continue
            wineries.add(winery)
            picks.append({"slug": slug, "card": self.cards[slug], "score": round(score, 4),
                          "reasons": reasons or ["подходит под ваши ответы"]})
        return {"total": len(picks), "picks": picks[offset:offset + limit], "note": " ".join(notes) or None}

    def _score(self, dish, color, sweetness, like, base) -> list[tuple[float, str, list[str], bool]]:
        """Colour and sweetness filter; dish fit and likeness only rank."""
        wanted = {"dry": {"сухое"}, "semi": {"полусухое", "полусладкое"}, "sweet": {"полусладкое", "сладкое"}}
        scored = []
        for slug, profile in self.profiles.items():
            card = self.cards[slug]
            if slug == like or card.get("reference_available") in (False, "False"):
                continue
            if color == "sparkling" and not profile.sparkling:
                continue
            if color in COLORS and (profile.color != color or profile.sparkling):
                continue
            score, reasons, strong = 0.0, [], not dish
            if sweetness:
                level = profile.sweetness or (None if profile.sparkling else "сухое")
                if level not in wanted[sweetness]:
                    continue
                reasons.append(level)
            if dish:
                pairing = recommend_pairings(card, dish)["selected"]
                fit = {"хорошо подходит": 3.0, "можно попробовать": 1.5}.get(pairing["fit"], 0.0)
                mentions = any(stem in card.get("description", "").lower() for stem in DISH_WORDS[dish])
                score += fit + 2.0 * mentions
                strong = fit >= 1.5 or mentions
                if mentions:
                    reasons.append(f"описание советует: {DISHES[dish].lower()}")
                elif fit:
                    reasons.append(f"{DISHES[dish].lower()}: {pairing['fit']}")
            if base is not None:
                text, _ = self._text_similarity(base, profile)
                union = base.grapes | profile.grapes
                common = sorted(base.grapes & profile.grapes)
                score += 3 * text + 2 * (len(common) / len(union) if union else 0)
                if common:
                    reasons.append("тот же сорт: " + ", ".join(common[:2]))
            # Prefer richer cards: a longer description gives the user more to read.
            score += min(len(card.get("description", "")), 600) / 1200
            scored.append((score, slug, reasons, strong))
        return scored
