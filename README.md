# Сканер российских вин

Покупатель фотографирует этикетку — сервис за 1–2 секунды находит **точную карточку вина** из каталога «Своё Вино» и показывает её в мобильном интерфейсе в стиле портала. Главная сложность задачи — двойники: одна и та же этикетка у вин разной сладости, сорта или года. Их различает локальная модель, которая читает этикетку.

Задача №10 РСХБ, хакатон «Лидеры цифровой трансформации» 2026. Команда «Коррозия ГПУ». Всё работает локально на одной видеокарте, без внешних API.

## Результаты

| Что | Значение |
|---|---|
| Top-1 на 62 вручную подтверждённых официальных фото | **60/62, F1@1 = 0,968** |
| Top-5 | **61/62 (0,984)** |
| Ответы с уверенностью ≥ 0,9 | 52 из 52 верны — экран выбора можно не показывать |
| Время ответа, RTX 3080 | p50 1,05 с, p95 1,21 с (SLA по ТЗ — до 3 с) |
| Без GPU | визуальный режим, ~1,2 с на фото |

Как размечали, где цифры могут быть завышены и как их воспроизвести — в [EVALUATION.md](EVALUATION.md).

## Как это работает

1. **Нормализация фото:** поворот по EXIF и три кропа — весь кадр, центр, широкий центр.
2. **Визуальный поиск:** SigLIP 2 сравнивает кропы с четырьмя видами каждого из 2 089 эталонов и отбирает 7 кандидатов.
3. **Проверка этикетки:** Qwen3.5-4B (4 бита) смотрит на фото и 7 эталонов, читает сорт, сладость и год и выбирает одну карточку.
4. **Ответ:** `slug`, уверенность и top-5 → API, карточка вина, похожие вина, сомелье, дневник.

Подробно — в [ARCHITECTURE.md](ARCHITECTURE.md).

## Что умеет интерфейс

- **Карточка вина** с процентом совпадения: регион, категория, сладость, сорт, описание, сочетания.
- **«Не то вино?»** — ближайшие карточки той же серии со сладостью. Если сканер не уверен, рядом крупно ваше фото, чтобы сверить этикетку.
- **Похожие вина других виноделен** — с объяснением: тот же сорт, та же сладость, регион.
- **Цифровой сомелье** — три вопроса (блюдо, цвет, сладость) и подборка из каталога.
- **Дневник** — оценка звёздами, заметка и ваше фото; хранится в браузере.

Подробно, со всеми эндпоинтами — в [docs/FEATURES_AND_API.md](docs/FEATURES_AND_API.md).

## Установка и запуск

Требуются Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), драйвер NVIDIA с поддержкой CUDA 12.8 (570+), GPU от 8 GiB, `7z`, около 15 GiB для архивов, эталонов и весов. `pyproject.toml` и `uv.lock` фиксируют зависимости для Linux x86-64: torch 2.11 (cu128), transformers 5.17, bitsandbytes 0.50. Команды ниже выполняются из корня репозитория. Исходные файлы положите в `data/source/Датасет.zip`, `data/source/official_real_photos.zip` и `data/field/real_photos.zip`. Последний архив содержит личные магазинные фото и нужен только для локальной проверки.

```bash
sudo apt-get install -y 7zip 7zip-rar jq
uv sync --locked --python /usr/bin/python3.12

uv run --locked python -m scanner.prepare
uv run --locked python -m scanner.reference_fixes
uv run --locked python -m scanner.model_fetch
uv run --locked python -m scanner.verifier download
uv run --locked python -m scanner.vision build --batch-size 16
uv run --locked python -m scanner.verifier thumbnails
uv run --locked python -m unittest scanner.test_catalog scanner.test_vision \
  scanner.test_ocr scanner.test_server scanner.test_evaluate scanner.test_score \
  scanner.test_pairing scanner.test_reference_fixes scanner.test_verifier scanner.test_recommend -v
uv run --locked uvicorn scanner.server:app --host 127.0.0.1 --port 8088 --workers 1
```

Сервис откроется на `http://127.0.0.1:8088/`, проверка готовности — `/health`. Что делает каждый шаг, сколько места занимают модели и как проверяются хеши — в [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

Оценка на размеченных официальных фото:

```bash
uv run --locked python -m scanner.evaluate --images-dir data/official_real_photos \
  --labels evaluation/official_labels_v2.tsv --output data/evaluation/v2_verifier.jsonl --use-verifier
```

`Dockerfile` содержит только окружение; `data/` готовится на хосте и монтируется. На проверочной машине образ пока не собирался:

```bash
docker build -t wine-scanner .
docker run --gpus all -p 8088:8088 -v "$PWD/data:/app/data" wine-scanner
```

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `WINE_VERIFIER` | `1` на GPU, `0` на CPU | `0` — только визуальный поиск, без Qwen; `1` на CPU — около минуты на фото |
| `WINE_OCR` | `0` | `1` — прежний EasyOCR-реранкер; действует только при `WINE_VERIFIER=0` |
| `WINE_DEVICE` | авто | `cpu` или `cuda` |
| `WINE_MANIFEST`, `WINE_INDEX` | исправленный каталог, `siglip2_multiview.npz` | другой каталог или индекс |
| `WINE_MODEL_DIR`, `WINE_VERIFIER_DIR`, `WINE_OCR_MODEL_DIR`, `WINE_IMAGES_DIR` | папки в `data/` | расположение весов и эталонов |

## Проверка скриптом организатора

```bash
curl -F image=@data/official_real_photos/48.98_02-09-2026_18-34-15.webp \
  http://127.0.0.1:8088/v1/eval/predict
```

Ответ — `{"slug":"denisov_rubin_klaret_krasnaya_strelka"}`. Контракт организатора принимает одно multipart-поле `image` и всегда возвращает один непустой `slug`, даже при низкой уверенности.

```bash
bash evaluation/organizer/participant_test.sh --images-dir data/source/eval/queries \
  --manifest data/source/eval/queries.tsv \
  --endpoint http://127.0.0.1:8088/v1/eval/predict \
  --output predictions.jsonl
```

Скрипт откажется перезаписать существующий `predictions.jsonl`: для повторной проверки укажите новый путь. В `data/source/eval/README.md` описан формат выдачи; правильные ответы для этих трёх фото организаторы не приложили.

## Ограничения

- **Оценка — на наборе разработки.** Число кандидатов и вес объединения подбирались на тех же 62 фото, поэтому 60/62 — оценка разработки, а не удержанного теста. Честное измерение — закрытый набор организаторов.
- **Вина вне каталога.** Сканер всегда возвращает ближайшую карточку. Уверенность не говорит, есть ли вино в каталоге: 18 из 35 фото вин вне каталога получают ≥ 0,7.
- **Одна бутылка в центре.** Снимок должен показывать целевую бутылку ближе к центру; для кадров с несколькими бутылками нужно отдельное правило оценки.
- **Общие эталоны.** 9 эталонных изображений привязаны сразу к двум `slug`, по картинке их не различить.
- **CPU.** Без видеокарты проверка этикетки отключена (иначе около минуты на фото), точность визуального режима ниже.
- **Docker.** `Dockerfile` содержит только окружение и на проверочной машине не собирался.

## Документация

| Файл | О чём |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | пайплайн и границы слоёв: нормализация, поиск, проверка, выдача карточки, доп. функции |
| [EVALUATION.md](EVALUATION.md) | разметка 100 официальных фото, метрики, ошибки, воспроизведение |
| [docs/FEATURES_AND_API.md](docs/FEATURES_AND_API.md) | интерфейс, функции после поиска, эндпоинты API |
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | подготовка данных, модели, хеши, исправленный каталог |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | что пробовали и почему отказались: варианты пайплайна, CPU, внешние датасеты, обученный каскад |
| [evaluation/](evaluation/) | разметка официальных фото, аудит каталога, съёмка полки, скрипт организатора |
