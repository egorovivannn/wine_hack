# Пилот независимых фотографий каталога, 24 сентября 2026

## Решение

Публичный поиск по 100 `slug` дал **одну** точную независимую фотографию после проверки этикетки и винтажа. Это студийный снимок бутылки, не полка. Нового train/validation/test набора нет; дообучение и переключение API не обоснованы. Действующий SigLIP2 + OCR остаётся в сервисе. 100 официальных и 13 собственных магазинных фото, включая их производные, в этом проходе не открывались, не оценивались и не использовались для выбора.

## Метод и выход

Выборка зафиксирована в [`manifests/catalog_photo_pilot_100.tsv`](manifests/catalog_photo_pilot_100.tsv): 25 визуально близких пар внутри разных производителей (50 карточек) по cosine этикеточных эталонов и 50 случайных карточек из остальных, seed `20260924`. Использованы только `data/catalog_manifest.json` и `data/index/siglip2.npz`. Поисковый запрос для каждой карточки: `"{name}" "{winery}" вино бутылка фото -site:vino-svoe.ru`. Поиск выполнен через публичную веб-выдачу; сырые результаты сохранены в игнорируемом `data/external/catalog_photo_pilot_search.json`. На один `slug` проверялись не более двух страниц с разных доменов, с паузой не менее секунды на домен и проверкой `robots.txt`; CAPTCHA не обходилась.

| Этап | Число |
| --- | ---: |
| Выбрано карточек / найдена хотя бы одна ссылка | 100 / 99 |
| Ссылок в поисковой выдаче | 742 |
| Попыток открыть страницу / HTTP 200 | 191 / 131 |
| Скачано изображений / разных SHA-256 | 59 / 52 |
| Ручной статус `verified` / `ambiguous` / `reject` | **1 / 25 / 33** |

Семь лишних строк среди 59 — идентичные исходные изображения, повторившиеся для разных `slug` или страниц. Прямых совпадений SHA с эталоном и pHash distance ≤ 8 не было; два визуально близких к эталонному товарному вырезу изображения отклонены как возможные преобразования исходного снимка. Для каждого скачанного файла сохранены URL страницы и изображения, UTC-время, SHA страницы/изображения/эталона, pHash distance, решение и причина в [`manifests/catalog_public_photo_review_2026-09-24.tsv`](manifests/catalog_public_photo_review_2026-09-24.tsv). Неоднозначные строки оставлены отдельным статусом, без скрытой разметки. Сырые ответы, фотографии и контактные листы — только в `data/external/`.

Подтверждённый пример: [`Mountain Eagle Aleatico Rose 2022`](https://www.cigarpro.ru/drinks/wine/russian-wines/mountain-eagle/agrolain-mountain-eagle-aleatico-rose-750-ml/) от Агролайн. Источник указывает 2022, розовое сухое, Алеатико; на эталонной этикетке читается 2022. Фотографии имеют разный ракурс, силуэт и отражения, SHA различается, pHash distance 24. Это отдельный студийный снимок, поэтому его пригодность для обучения на полочные сцены низкая. Действующий API вернул правильный `slug` на этой **одной** паре; 1/1 — качественный smoke check, не оценка точности.

Отказы иллюстрируют риск автоматической разметки: [Chateau Andre Gruner](https://www.osteria.ru/wine/russia/chateau-andre-gruner/) на странице имеет 2023 при 2022 на эталонной этикетке; [Chateau Tamagne Reserve](https://bestwine24.ru/vino-chateau-tamagne-reserve-premier-blanc-limited-edition-2020-075-l/) имеет 2020 против явного 2021 в одном из `slug`. У Legato Sauvignon Blanc и Sauvignon Blanc Mtsvane, а также у двух вариантов Villa Urkusta Cabernet Sauvignon один и тот же SHA эталона назначен двум `slug`; внешняя фотография не устраняет эту неоднозначность. Десятки найденных URL вели к баннерам, виноградникам, неподходящим бутылкам или страницам без доступного изображения.

При одном подтверждённом товаре невозможно сделать независимый split по продукту, источнику или съёмке и невозможно измерить статистическую Top-1/Hit@5. Пилот ограничен 100 карточками и первыми двумя разными доменами; оценка выхода относится только к этому способу поиска, а не ко всему интернету. Продолжать широкий скрейпинг тем же способом до дедлайна нецелесообразно.

## Retrieval и решение по API

На прежней **внешней validation**, по тем же сохранённым признакам, отдельно измерен выход правильного товара из Top-20 замороженного энкодера и CE-адаптера. Объединение означает максимум 40 кандидатов без отбрасывания baseline Top-20; это анализ candidate recall, не новая версия сервиса.

| Validation | Frozen Top-1 / Hit@5 / Recall@20 | CE Top-1 / Hit@5 / Recall@20 | Union Recall@≤40 |
| --- | ---: | ---: | ---: |
| WineSensed, 864 query | 692 / 822 / 844 | 728 / 839 / 855 | 856 |
| Norwegian Grocery, 317 query | 253 / 299 / 308 | 260 / 297 / 306 | 310 |

На Norwegian CE потерял четыре baseline Top-20 кандидата и добавил два. Поэтому адаптер без сохранения baseline-кандидатов не включён. Полный вид, детекторный crop и OCR-текст на реальных **российских** фото сравнить статистически нельзя: найденная точная пара одна и она студийная. Широкий подбор OCR-порогов или новых heads по ней запрещён этим протоколом. Внешний test повторно не запускался для выбора конфигурации. API, включая JSON-контракт, не изменён.

## Воспроизведение и проверка

```bash
uv sync --locked --python /usr/bin/python3.12 --group audit --group train
uv run --locked --group audit python -m evaluation.catalog_photo_pilot
# Требуется сохранённый поисковый JSON из пилота; повторная загрузка страниц создаёт новый snapshot.
uv run --locked --group audit python -m evaluation.audit_public_photos
uv run --locked --group audit python -m evaluation.public_photo_contact_sheet
uv run --locked --group audit python -m evaluation.finalize_public_photo_review
uv run --locked python -m evaluation.analyze_candidate_union \
  --views data/external/external_v3/views.jsonl \
  --features data/external/external_v3/features.npz \
  --head data/models/training/external_adapter_ce_v3/best.pt \
  --split val --output data/evaluation/candidate_union_val_2026-09-24.json
```

`audit_public_photos` отказывается перезаписать существующий raw JSONL. Сайты могут измениться; сохранённые SHA фиксируют именно этот снимок. Оригинальные веб-фотографии и эталоны остаются вне Git. Ручные решения фиксируются в [`manifests/catalog_public_photo_decisions_2026-09-24.tsv`](manifests/catalog_public_photo_decisions_2026-09-24.tsv).

SHA-256: выборка `efb096d88ec30bd204878530026d249d946d5f82be190a50ec01e9daf28c26f3`; сырая выдача `0820032a5a4a8a1099a249c88a1718063055aa982f6470d33d21f6fced8aa0db`; аудит `e2304abb08299067ca9b53c86ef6c7264e41422e99d24d0225201846cee8c448`; решения `2b37424229d6cecd700e38a30c63294dc9a402f3935860b5935396bdee6b970e`; ревью `1921af78238d7c6e203d6be81868ea2895e70d0e6349ece6ab151f34b6463e1d`; каталог `5654d67ef71dfd3462aad1a00d949f5f5465a4c30f63667b95f388d68289a21d`; индекс `88af9e5dabefbcceb02b24b36cf4c31d8e46e36cea3de5363c6acf7b10c4bd4f`; SigLIP2 `ed72c0ace85020ae610fc817c2538b9cae5a477b012a50859c60af5b3ad30857`; CE-head `b6b3b64634ae6342e1ef51ad6d5788bc90f5711205f814cec95bc06497ff1420`. Устройство: RTX 3080 (10 GiB), Ryzen 5 5600, ≈55 GiB доступной RAM до запуска; Python 3.12.3.

Из нового `uv`-окружения пройдены 31 unit test, Ruff E/F/I и format. С `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` локально загружены `/`, `/health`, `/v1/predict` и `/v1/eval/predict`; последний вернул ровно непустой `slug` на подтверждённом публичном снимке. В `index.html` нет внешних JS/CSS URL. Официальный benchmark не запускался, поскольку финальная модель не заморожена и у этого этапа нет улучшения, требующего единственной holdout-проверки.
