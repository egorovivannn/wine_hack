"""Deterministic food-pairing guidance from the catalog's limited wine metadata."""

from __future__ import annotations

DISHES = {
    "fish": "Рыба",
    "seafood": "Морепродукты",
    "poultry": "Птица",
    "red_meat": "Красное мясо",
    "vegetables": "Овощные блюда",
    "cheese": "Сыр",
    "dessert": "Десерт",
    "spicy": "Острые блюда",
}

STYLE = {
    "white": (
        "Белое вино",
        ("fish", "seafood", "poultry"),
    ),
    "red": (
        "Красное вино",
        ("red_meat", "cheese", "poultry"),
    ),
    "rose": (
        "Розовое вино",
        ("poultry", "vegetables", "cheese"),
    ),
    "orange": (
        "Оранжевое вино",
        ("cheese", "poultry", "vegetables"),
    ),
    "sparkling": (
        "Игристое вино",
        ("seafood", "fish", "cheese"),
    ),
    "off_dry": (
        "Полусладкое вино",
        ("spicy", "cheese", "dessert"),
    ),
    "sweet": (
        "Сладкое вино",
        ("dessert", "cheese", "spicy"),
    ),
}


PAIR_REASONS = {
    "white": {
        "fish": "Свежий профиль белого вина обычно подходит к нежному вкусу рыбы.",
        "seafood": "Свежесть вина не перекрывает тонкий вкус морепродуктов.",
        "poultry": "Лёгкая структура вина подходит к блюдам из птицы без тяжёлого соуса.",
    },
    "red": {
        "red_meat": "Более плотный профиль красного вина подходит к насыщенному мясному блюду.",
        "cheese": "Выразительный сыр может поддержать насыщенность красного вина.",
        "poultry": "Птица с насыщенным соусом может сочетаться с красным вином.",
    },
    "rose": {
        "poultry": "Лёгкий ягодный профиль подходит к птице без тяжёлого соуса.",
        "vegetables": "Розовое вино может поддержать вкус овощей на гриле.",
        "cheese": "Не слишком выдержанный сыр сохраняет баланс с лёгким розовым вином.",
    },
    "orange": {
        "cheese": "Насыщенный аромат оранжевого вина подходит к выразительному сыру.",
        "poultry": "Структура вина может выдержать птицу с пряностями.",
        "vegetables": "Запечённые овощи поддерживают насыщенный профиль вина.",
    },
    "sparkling": {
        "seafood": "Пузырьки освежают вкус после морепродуктов.",
        "fish": "Игристое вино не перекрывает нежный вкус рыбы.",
        "cheese": "Свежесть игристого вина помогает уравновесить солоноватый сыр.",
    },
    "off_dry": {
        "spicy": "Умеренная сладость может смягчить остроту блюда.",
        "cheese": "Солоноватый сыр может уравновесить сладость вина.",
        "dessert": "К умеренно сладкому десерту можно попробовать полусладкое вино.",
    },
    "sweet": {
        "dessert": "Сладкое вино обычно подают к десерту сопоставимой сладости.",
        "cheese": "Контраст сладкого вина и солоноватого сыра может быть удачным.",
        "spicy": "Сладость вина может смягчить остроту блюда.",
    },
}


def _style(card: dict) -> str:
    # The description contains tasting notes such as "sweet berries"; infer sweetness
    # only from the product's own name or slug.
    title = (card.get("name", "") + " " + card.get("slug", "")).lower()
    if "полуслад" in title:
        return "off_dry"
    if "сладкое" in title or "sladkoe" in title:
        return "sweet"
    if any(
        token in title
        for token in ("брют", "bryut", "brut", "игрист", "pet-nat", "петнат")
    ):
        return "sparkling"
    category = card.get("category", "").lower()
    if "красн" in category:
        return "red"
    if "розов" in category:
        return "rose"
    if "оранж" in category:
        return "orange"
    return "white"


def _reason(card: dict, style: str, dish: str) -> str:
    reason = PAIR_REASONS[style].get(
        dish, "По данным карточки нельзя уверенно оценить это сочетание."
    )
    description = card.get("description", "").lower()
    if dish in {"fish", "seafood"} and "кислотн" in description:
        reason += " В описании отмечена кислотность, которая освежает вкус."
    if dish == "red_meat" and "танин" in description:
        reason += " В описании отмечены танины, которые поддерживают мясное блюдо."
    return reason


def recommend_pairings(card: dict, dish: str | None = None) -> dict:
    """Return transparent suggestions; no per-bottle expert claim is implied."""
    if dish is not None and dish not in DISHES:
        raise ValueError(f"Unknown dish: {dish}")
    style = _style(card)
    style_label, suggested = STYLE[style]
    result = {
        "slug": card["slug"],
        "style": style_label,
        "basis": "Подсказка основана на категории, названии и отдельных сведениях из описания вина; вкусовые предпочтения могут отличаться.",
        "suggestions": [
            {"dish": key, "label": DISHES[key], "reason": _reason(card, style, key)}
            for key in suggested
        ],
        "selected": None,
    }
    if dish is not None:
        fit = (
            "хорошо подходит"
            if dish in suggested[:2]
            else (
                "можно попробовать"
                if dish == suggested[2]
                else "сочетание зависит от блюда"
            )
        )
        reason = _reason(card, style, dish)
        if dish not in suggested:
            reason += f" Для более предсказуемого сочетания рассмотрите вариант: {DISHES[suggested[0]].lower()}."
        result["selected"] = {
            "dish": dish,
            "label": DISHES[dish],
            "fit": fit,
            "reason": reason,
        }
    return result
