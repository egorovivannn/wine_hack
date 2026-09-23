# Сканер российских вин

Локальный сервис получает фотографию бутылки и возвращает точный `slug` карточки из каталога конкурса. Мобильная страница показывает найденное вино, визуально похожие варианты и сохраняет заметку в дневнике браузера. Рабочая реализация находится в ветке `wine-scanner-development`.

## Проверенное состояние

- Индекс: 2 087 уникальных эталонных изображений, связанных с 2 098 карточками. Всего в `df_2.csv` 2 103 уникальных `slug`; у пяти нет пригодного эталона.
- Визуальная модель: `google/siglip2-base-patch16-384`, ревизия `f775b65a79762255128c981547af89addcfe0f88`. Для близких вариантов дополнительно используется EasyOCR 1.7.2. Веса обеих моделей и индекс проверяются по SHA-256 при запуске.
- На RTX 3080 (10 GiB) индекс построен за 169 секунд. Повторный прогон 100 официальных снимков с OCR: p50 690 мс, p95 1 054 мс; OCR понадобился на 51 кадре. Без OCR: p50 388 мс, p95 426 мс. Протокол и хеши — в [evaluation/RESULTS.md](evaluation/RESULTS.md).
- На **17 вручную отобранных и подтверждённых** снимках Top-1 = 16/17 с OCR против 14/17 без OCR. На семи подтверждённых фото других дней обе версии дали 7/7. Эти малые и смещённые выборки не являются оценкой скрытого набора; остальные официальные фото не имеют опубликованных ответов.
- Скрипт организатора `participant_test.sh` успешно обработал три контрольных запроса через HTTP.

## Воспроизведение на Ubuntu 24.04 с NVIDIA GPU

Требуются Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), драйвер NVIDIA с поддержкой CUDA 12.4, `7z`, свободное место для архивов и эталонов. `pyproject.toml` и `uv.lock` фиксируют зависимости для Linux x86-64 с CUDA 12.4. Команды ниже выполняются из корня репозитория. Исходные файлы положите в `data/source/Датасет.zip`, `data/source/official_real_photos.zip` и `data/field/real_photos.zip`. Последний архив содержит личные магазинные фото и нужен только для локальной проверки.

```bash
sudo apt-get install -y 7zip 7zip-rar jq
uv sync --locked --python /usr/bin/python3.12

uv run --locked python -m scanner.prepare
uv run --locked python -m scanner.model_fetch
uv run --locked python -m scanner.ocr download
uv run --locked python -m scanner.vision build --batch-size 16
uv run --locked python -m unittest scanner.test_catalog scanner.test_vision \
  scanner.test_ocr scanner.test_server scanner.test_evaluate scanner.test_score \
  embedding.test_pipeline -v
uv run --locked uvicorn scanner.server:app --host 127.0.0.1 --port 8088 --workers 1
```

Скрипт подготовки проверяет SHA-256 всех трёх исходных архивов и `df_2.csv`, извлекает ровно 2 087 нужных файлов из многотомного RAR, 100 официальных и 13 собственных фото. Он не использует порядок файлов в архиве или числовые префиксы имён фото как метки. При повторном запуске проверенные файлы не распаковываются заново. Только проверить подготовленные данные:

```bash
uv run --locked python -m scanner.prepare --verify-only
```

Все исходные архивы, фото, веса и индекс находятся в игнорируемом `data/`. Папка `data/models/siglip2-base-patch16-384` занимает около 1,4 GiB; два OCR-веса в `data/models/easyocr` — около 99 MiB. Индекс: `data/index/siglip2.npz` и `data/index/siglip2.json`. `scanner.model_fetch` загружает только закреплённую ревизию модели; `scanner.ocr download` проверяет SHA-256 загруженных весов; `scanner.vision build` отклоняет изменение эталонов относительно манифеста. Для запуска только визуальной версии задайте `WINE_OCR=0` перед `uvicorn`.

## API и проверка организатора

```bash
curl -F image=@data/official_real_photos/48.98_02-09-2026_18-34-15.webp \
  http://127.0.0.1:8088/v1/eval/predict
```

Ответ `{"slug":"denisov_rubin_klaret_krasnaya_strelka"}`. Контракт организатора принимает одно multipart-поле `image` и один непустой `slug`. Полная карточка и альтернативы доступны по `POST /v1/predict`; `GET /v1/wines/{slug}` возвращает карточку. Страница сканера находится по `/`, здоровье сервиса — `/health`.

```bash
cd data/source/eval
bash participant_test.sh --images-dir queries --manifest queries.tsv \
  --endpoint http://127.0.0.1:8088/v1/eval/predict \
  --output predictions.jsonl
```

Скрипт откажется перезаписать существующий `predictions.jsonl`: для повторной проверки укажите новый путь. В `data/source/eval/README.md` описан формат выдачи; правильные ответы для этих трёх фото организаторы не приложили.

## Локальная оценка

`evaluation/manual_official_labels.tsv` содержит вручную просмотренные снимки с SHA-256, видимым доказательством и состоянием `verified`, `unknown` или `ambiguous`. Состояния `unknown` и `ambiguous` не включаются в знаменатель точности.

```bash
uv run --locked python -m scanner.evaluate \
  --images-dir data/official_real_photos \
  --labels evaluation/manual_official_labels.tsv \
  --output data/evaluation/official_predictions.jsonl --use-ocr
```

Рядом создаётся `data/evaluation/official_predictions.summary.json` с точностью на подтверждённых примерах, p50/p95 задержки, версиями ПО, именем GPU, ревизией Git и хешами моделей, индекса, манифеста и меток. Для сравнения без OCR повторите команду с другим выходным файлом и без `--use-ocr`. Одни и те же сохранённые предсказания можно проверить по независимой разметке без повторного GPU-прогона:

```bash
uv run --locked python -m scanner.score \
  --predictions data/evaluation/official_predictions.jsonl \
  --labels evaluation/holdout_official_labels.tsv
```

Метрика удобна для регрессий на этих примерах; для выбора следующей модели нужна более крупная удержанная группа снимков по сеансам съёмки.

## Известные ограничения и следующий шаг

Снимок должен показывать целевую бутылку ближе к центру. OCR исправил два различия внутри серий на отладочных снимках, но малый удержанный набор не доказывает прирост на скрытом тесте. 11 эталонных файлов в источнике привязаны сразу к двум `slug`, поэтому изображение не всегда позволяет выбрать точную карточку. Фотографии неизвестного вина и кадры с несколькими бутылками требуют отдельного правила оценки от организаторов. Подробности — в [ARCHITECTURE.md](ARCHITECTURE.md).
