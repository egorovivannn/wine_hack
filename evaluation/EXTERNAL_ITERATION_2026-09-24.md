# Внешняя итерация распознавания, 24 сентября 2026

## Решение

Новый YOLO26s + SigLIP2 residual adapter + Top-20 reranker **не подключён к API**. На независимых внешних фото одного и того же вина адаптер улучшил поиск, но на полочных товарах прирост Top-1 составил 0–2 из 206 запросов, Recall@20 снизился на два запроса, а OCR не улучшил validation относительно визуального реранкера. Перенос на российские винные полки этим опытом не доказан. Скрытый тест организаторов остаётся единственным честным конкурсным тестом.

Все 100 официальных фото и 13 собственных магазинных фото были изолированы: в этой итерации их изображения, кропы, признаки, OCR, метки, выдача и метрики не использовались для подготовки данных, подбора, обучения, отладки или оценки. Старый отчёт о цепочке прочитан только для понимания архитектуры. Новый официальный benchmark не запускался; прежние 59 примеров уже не слепые.

## Данные и границы независимости

| Источник | Реальные query / gallery | Проверка соответствия | Разделение |
| --- | ---: | --- | --- |
| [WineSensed, Figshare article 23376560](https://api.figshare.com/v2/articles/23376560), `chunk_001.zip` | 6 842 query + 2 284 reference, по одному reference и трём query на vintage | `vintage_id` из опубликованного metadata.zip; 9 136 запрошенных JPEG, 9 129 уникальных после скачивания, 9 126 после аудита близких дублей | Vintage целиком в одном split. Известные winery ID и pHash-связанные классы вместе. Все 300 ранее оценённых vintages принудительно оставлены только в train. Train/val/test: 1 693/288/303 класса и 5 070/864/908 query. У 2 208 классов winery ID неизвестен; capture-session ID нет. |
| [Norwegian Grocery](https://huggingface.co/datasets/valiantlynxz/norwegian-grocery), revision `17cacfcb3ed0a0a24e757fa92c5c546cb3a19f57`, CC-BY-NC-4.0 | 4 501 query + 319 reference; train/val/test query 3 978/317/206 | В опубликованном COCO нет заявленного `product_code` у каждого bbox: приняты только однозначные точные category-name → product metadata joins с существующим reference. 319 товаров, 11 661 исходных bbox-пар. Один bbox на товар/сцену, выбирается самый крупный. | Product и pHash-близкие reference не расходятся между split. Соседние файлы, сцены с высоким совпадением категорий и pHash-близкие сцены объединены: 15 компонентов. Train/val/test: 177/37/30 использованных сцен. Из-за отсутствия session ID независимость сеансов доказать нельзя. |
| [GRAIN](https://zenodo.org/records/16410628), независимый detector test | 273 кадра, 410 рамок | IoU≥0,5 с размеченной этикеткой, ближайшей к центру | Отдельный опубликованный detector test; он не использовался для нового обучения детектора. |

WineSensed query: декодирование с ориентацией, тот же YOLO26s и селектор центральной этикетки, затем белый квадрат 384×384. При отсутствии рамки применяется прежний центральный fallback. Reference: фиксированный вид этикетки `scanner.image.reference_views(...)[1]`. WineSensed не содержит полочного контекста; его кадры не объявляются полочными bbox. Norwegian query: **истинный bbox товара** из COCO, crop с 4% полем и белым квадратом 384×384; reference: полное фото товара `reference_views(...)[0]`. Таким образом Norwegian retrieval проверяет сопоставление полочного кропа со студийным фото, но не проверяет автоматическую локализацию.

WineSensed: 99/6 842 query (1,45%) прошли по fallback без рамки; отсутствие GT bbox не позволяет считать остальные локализации верными. На GRAIN независимая проверка селектора дала хотя бы одну рамку в 267/273 (97,8%), центральную рамку с IoU≥0,5 в 263/273 (96,3%), три кадра без предсказания; p50/p95 детектора 14,7/23,1 мс. Эта метрика проверяет геометрию размеченной этикетки у центра, а не идентичность нужной бутылки в магазине. Поэтому детектор повторно не обучался.

GRAIN test отделён от train/val детектора, но на нём уже измеряли прежний mAP. Новых настроек детектора по этому test не выбирали; считать его полностью слепым нельзя.

## Обучение и выбор

Замороженный `google/siglip2-base-patch16-384` выдаёт 768-мерный нормированный признак для каждого query/reference view. Обучается только компактная residual-голова `LabelHead` на NVIDIA RTX 3080 10 GiB. Для каждого реального train query положительный reference имеет тот же verified identity; 63 отрицательных — ближайшие другие **train** references по исходному SigLIP2. pHash-близкие продукты из одной ambiguity group исключены как отрицательные. Сравнены только CE и margin loss, по 12 эпох, seed 42; ранняя выбранная эпоха определяется 0,4 × WineSensed val Top-1 + 0,6 × Norwegian val Top-1. CE: лучший epoch 9, score 0,82915, Wine 728/864, Norwegian 260/317. Margin: epoch 1, score 0,81514, Wine 710/864, Norwegian 257/317. Пик памяти головы 0,214 GiB, 7,4 с обучения CE; этап извлечения признаков отдельно занял 73,3 с для WineSensed и 54,3 с для Norwegian, пик VRAM 0,817 GiB. Доступная GPU-память до обучения была ≈5,9 GiB, RAM ≈55 GiB, диск ≈111 GiB.

Перед дорогими прогонами 24 сентября около 17:50 МСК до заявленного дедлайна 29 сентября 23:59 МСК оставалось примерно 5 дней 6 часов. Поэтому полный большой grid search не запускался; два loss и один линейный реранкер укладывались в доступные ресурсы и оставляли время на аудит.

Отдельно построено 57 935 кандидатных пар «похожие товары» среди 2 087 каталожных эталонов по визуальному косинусу, OCR и метаданным. 2 001 пар помечены как неоднозначные (включая общие изображения/одинаковые нормализованные имена). Остальные 55 934 **не подтверждены как разные wine identities** и не попали в supervised loss: к этим эталонам нет проверенных реальных positive query, а синонимы и версии этикеток могут дать false negative. Каталожные пары сохранены вне Git для последующего ручного аудита; они не используются как скрытая разметка внешнего test.

Top-20 реранкеры обучены только на Norwegian train query и разрешённой train gallery. Visual-only использует исходный/адаптированный cosine, относительный score и rank. Visual+OCR добавляет IDF-взвешенное совпадение OCR-слов с опубликованным названием продукта; OCR-кэш привязан к SHA кропа. Обе модели получили лучший validation Top-1 262/317 на epoch 48, а OCR коэффициент при этом отрицателен. При равенстве выбран visual-only как более простой. Сравнение с текущим hybrid является **proxy**: вызывается действующий `OCRReranker` с его порогом, но из доступных полей карточки есть лишь название норвежского товара. Это не полный российский API.

## Абляция: одинаковые query, gallery и split внутри каждого блока

Метрики в ячейках: Top-1 / Hit@5 / Recall@20. Gallery: Wine val 1 981 и test 2 284; Norwegian val 270 и test 319. Validation содержит 864 Wine и 317 Norwegian query, test — 908 и 206. Test запущен один раз после выбора CE и checkpoints по validation; после просмотра test модели не дообучались.

Аудит прежних подготовленных Norwegian views показал нулевое пересечение 206 финальных test annotation ID с прежними train/val (старые test ID не оценивались). Это дополнительно исключает использование пилотной validation как скрытого сигнала финального test.

| Данные | Метод | Validation | Test |
| --- | --- | --- | --- |
| WineSensed | Frozen SigLIP2 | 692 / 822 / 844 | 722 / 873 / 887 |
| WineSensed | CE adapter | **728 / 839 / 855** | **758 / 879 / 893** |
| Norwegian Grocery | Frozen SigLIP2 | 253 / 299 / 308 | 129 / 182 / **201** |
| Norwegian Grocery | Текущий hybrid proxy | 251 / 299 / 308 | 129 / 182 / **201** |
| Norwegian Grocery | CE adapter | 260 / 297 / 306 | 129 / **194** / 199 |
| Norwegian Grocery | CE + visual reranker | **262 / 299 / 306** | 130 / 192 / 199 |
| Norwegian Grocery | CE + OCR reranker | **262 / 299 / 306** | **131 / 192 / 199** |

Wine test Top-1: 79,5% → 83,5%; Norwegian test: 62,6% → 63,1% для выбранного visual reranker или 63,6% для OCR ablation. На полочном test прирост 1–2 query при 206 запросах слишком мал, а recall@20 201 → 199 показывает регрессию поиска кандидатов. Каталожные российские вина и норвежские бакалейные товары различаются упаковкой и текстом; эти числа нельзя называть конкурсной точностью.

Типичные внешние ошибки по сохранённой test выдаче: `EVERGOOD CLASSIC KAFFEKAPSEL 16STK` ↔ `10STK`, `SJOKOLADEDRIKK REFILL 512G` ↔ обычный `512G`, `SOLEGG 6STK` ↔ `12STK`, `NESCAFE GOLD ORGANIC&FAIRTRADE 100G` ↔ `NESCAFE GULL 100G/200G`. Это близкие серии или размеры, требующие чтения мелкого текста. Для WineSensed отсутствие полной информации о winery и названиях в выбранном metadata-срезе не позволяет честно разбить ошибки на «тот же производитель/другой vintage»; сохранены identity и truth rank для каждого промаха.

## Задержка и контроль версий

Последовательный прогретый процесс, первые 100 test query каждого источника, RTX 3080; p50/p95 в мс:

| Путь | p50 | p95 | Что включено |
| --- | ---: | ---: | --- |
| Wine frozen visual | 43,6 | 55,4 | Чтение/хеш/декодирование, YOLO crop, SigLIP2, поиск по 2 284 refs |
| Wine adapter visual | 53,5 | 69,7 | То же + residual head и адаптированный поиск |
| Norwegian frozen visual | 15,0 | 19,7 | Чтение готового **oracle crop**, SigLIP2, поиск по 319 refs |
| Norwegian adapter visual | 15,4 | 20,1 | То же + residual head |
| Norwegian adapter + OCR scorer | 46,2 | 81,1 | То же + EasyOCR на каждом query и Top-20 scorer |

Измерение Norwegian исключает поиск bbox; измерение Wine использует один crop вместо двух видов текущего API. Поэтому эти задержки не являются измерением новой версии API. OCR на всех Norwegian query — верхняя оценка для сценария без порога запуска. Протокол и стадии — `data/evaluation/external_latency_v4.json`.

Machine-readable маленький манифест хешей: [`manifests/external_sources_2026-09-24.json`](manifests/external_sources_2026-09-24.json). Сырые JPEG, OCR, пары, признаки, checkpoints и подробные ошибки лежат только в игнорируемом `data/`. Основные SHA-256: combined views `763c033002aee21567faf40cdc20a53e80d4b1eedcdad42c48f8bd2012e7dc52`; features `787f0b55adab581d1fa48878359d722d1880af15354b4a7a30aa0b549e15987e`; YOLO `67b4c43518a07d8aa505b906296217e541779af0a488c4d92330ea3425087104`; frozen SigLIP2 `ed72c0ace85020ae610fc817c2538b9cae5a477b012a50859c60af5b3ad30857`; CE `b6b3b64634ae6342e1ef51ad6d5788bc90f5711205f814cec95bc06497ff1420`; visual reranker `data/models/training/external_reranker_v3/visual.pt`. Обучение и retrieval test выполнялись на ревизии `cacba8a159d404cfeab1af199d9f8bb0ba77f322`; исправленный latency scorer — `07b1221`. Python 3.12.3, PyTorch 2.6.0+cu124. Полные SHA отчётов и второго реранкера записаны в манифесте.

Figshare сообщил MD5 и размер полного `chunk_001.zip`, но полный архив локально не скачивался, поэтому его MD5 не пересчитывался. Каждый прочитанный ZIP member проверен CRC и SHA-256; metadata.zip проверен полным SHA-256. Версия Norwegian и исходные файлы зафиксированы SHA-256. Недоступные session ID и возможные семантические дубли за пределами pHash остаются ограничениями split.

Проверено поле WineSensed `experiment_id`: оно заполнено только у тех же 76 vintages, для которых есть winery ID, и каждый такой ID встречается внутри одного vintage. Среди этих ID нет пересечения split. Для остальных 2 208 vintages поле пустое, а смысл «сеанс съёмки» источником не установлен; использовать его как полный session key было бы необоснованно.

## Воспроизведение из корня репозитория

Команды ниже создают новые каталоги и откажутся перезаписать существующий результат. Для полного повтора используйте свободные версии путей вместо `v3/v4`; `uv` берёт закреплённый lock. Каталожный baseline и YOLO checkpoint готовятся по основному README и предыдущему [отчёту](TRAINED_CASCADE_2026-09-24.md).

```bash
uv sync --locked --group train
uv run --locked python -m embedding.prepare_winesensed_sample --classes 2284 --images-per-class 4 --seed 42 --output data/external/winesensed/sample_v3_raw
uv run --locked python -m embedding.clean_winesensed --input data/external/winesensed/sample_v3_raw/manifest.csv --prior-manifest evaluation/manifests/winesensed_prior_300_2026-09-24.csv --output data/external/winesensed/sample_v3_clean
uv run --locked python -m embedding.prepare_norwegian --output data/external/norwegian_v3
uv run --locked --group train python -m embedding.external_views --wine-manifest data/external/winesensed/sample_v3_clean/manifest.csv --detector data/models/training/detector_yolo26s_v2/weights/best.pt --output data/external/wine_views_v3
uv run --locked --group train python -m embedding.external_views --norwegian-root data/external/norwegian_v3 --output data/external/norwegian_views_v4
uv run --locked python -m embedding.encode_external_views --views data/external/wine_views_v3/views.jsonl --output data/external/wine_features_v3.npz --batch 8
uv run --locked python -m embedding.encode_external_views --views data/external/norwegian_views_v4/views.jsonl --output data/external/norwegian_features_v4.npz --batch 8
uv run --locked python -m embedding.combine_external --wine-views data/external/wine_views_v3 --wine-features data/external/wine_features_v3.npz --norwegian-views data/external/norwegian_views_v4 --norwegian-features data/external/norwegian_features_v4.npz --output data/external/external_v3
uv run --locked python -m embedding.ocr_external --views data/external/norwegian_views_v4/views.jsonl --output data/external/norwegian_ocr_v4.jsonl
uv run --locked python -m embedding.train_external_adapter --views data/external/external_v3/views.jsonl --features data/external/external_v3/features.npz --loss ce --epochs 12 --seed 42 --output data/models/training/external_adapter_ce_v3
uv run --locked python -m embedding.train_external_adapter --views data/external/external_v3/views.jsonl --features data/external/external_v3/features.npz --loss margin --epochs 12 --seed 42 --output data/models/training/external_adapter_margin_v3
uv run --locked python -m embedding.train_external_reranker --views data/external/external_v3/views.jsonl --features data/external/external_v3/features.npz --head data/models/training/external_adapter_ce_v3/best.pt --ocr data/external/norwegian_ocr_v4.jsonl --output data/models/training/external_reranker_v3
uv run --locked python -m evaluation.eval_external_retrieval --views data/external/external_v3/views.jsonl --features data/external/external_v3/features.npz --head data/models/training/external_adapter_ce_v3/best.pt --reranker-dir data/models/training/external_reranker_v3 --ocr data/external/norwegian_ocr_v4.jsonl --wine-manifest data/external/winesensed/sample_v3_clean/manifest.csv --norwegian-pairs data/external/norwegian_v3/pairs.jsonl --detector data/models/training/detector_yolo26s_v2/weights/best.pt --output data/evaluation/external_retrieval_v3.json
uv run --locked --group train python -m evaluation.latency_external --views data/external/external_v3/views.jsonl --features data/external/external_v3/features.npz --wine-manifest data/external/winesensed/sample_v3_clean/manifest.csv --detector data/models/training/detector_yolo26s_v2/weights/best.pt --head data/models/training/external_adapter_ce_v3/best.pt --reranker data/models/training/external_reranker_v3/ocr.pt --output data/evaluation/external_latency_v4.json
uv run --locked --group train python -m evaluation.eval_external_detector --data data/external/grain/yolo_v2/data.yaml --checkpoint data/models/training/detector_yolo26s_v2/weights/best.pt --output data/evaluation/external_detector_v4.json
```

Для каталожного mining после сборки эталонного индекса: `uv run --locked python -m embedding.mine_catalog_negatives --output data/external/catalog_hard_pairs_v4.jsonl`. Проверка тестов: `uv run --locked --group train python -m unittest embedding.test_external embedding.test_pipeline scanner.test_vision scanner.test_ocr -q`. Ruff `check --select E,F,I --ignore E501` и `format --check` выполнены для 14 новых и изменённых Python-файлов этой итерации; старые модули содержат существующие замечания Ruff и в этот проход не переписывались. Официальные фото в этом протоколе отсутствуют.
