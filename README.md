# Сканер российских вин

Локальный сервис получает фотографию бутылки и возвращает точный `slug` карточки из каталога конкурса. Мобильная страница показывает найденное вино, визуально похожие варианты и сохраняет заметку в дневнике браузера. Рабочая реализация находится в ветке `wine-scanner-development`.

## Проверенное состояние

- Индекс: 2 087 уникальных эталонных изображений, связанных с 2 098 карточками. Всего в `df_2.csv` 2 103 уникальных `slug`; у пяти нет пригодного эталона.
- Модель: `google/siglip2-base-patch16-384`, ревизия `f775b65a79762255128c981547af89addcfe0f88`. Вес и индекс проверяются по SHA-256 при запуске.
- На RTX 3080 (10 GiB) индекс построен за 169 секунд; одиночное фото обрабатывается примерно за 0,4 секунды после прогрева. Полный прогон 100 официальных снимков: p50 399 мс, p95 437 мс.
- На **17 вручную отобранных и подтверждённых** снимках Top-1 = 14/17, Top-5 = 16/17. Эта малая и смещённая выборка не является оценкой скрытого набора. Остальные официальные фото не имеют опубликованных ответов.
- Скрипт организатора `participant_test.sh` успешно обработал три контрольных запроса через HTTP.

## Воспроизведение на Ubuntu 24.04 с NVIDIA GPU

Требуются Python 3.12, драйвер NVIDIA с поддержкой CUDA 12.4, `7z`, свободное место для архивов и эталонов. Команды ниже выполняются из корня репозитория. Исходные файлы положите в `data/source/Датасет.zip`, `data/source/official_real_photos.zip` и `data/field/real_photos.zip`. Последний архив содержит личные магазинные фото и нужен только для локальной проверки.

```bash
sudo apt-get install -y 7zip 7zip-rar jq
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements-scanner.txt

.venv/bin/python -m scanner.prepare
.venv/bin/python -m scanner.model_fetch
.venv/bin/python -m scanner.vision build --batch-size 16
.venv/bin/python -m unittest scanner.test_catalog scanner.test_vision \
  scanner.test_server scanner.test_evaluate embedding.test_pipeline -v
.venv/bin/uvicorn scanner.server:app --host 127.0.0.1 --port 8088 --workers 1
```

Скрипт подготовки проверяет SHA-256 всех трёх исходных архивов и `df_2.csv`, извлекает ровно 2 087 нужных файлов из многотомного RAR, 100 официальных и 13 собственных фото. Он не использует порядок файлов в архиве или числовые префиксы имён фото как метки. При повторном запуске проверенные файлы не распаковываются заново. Только проверить подготовленные данные:

```bash
.venv/bin/python -m scanner.prepare --verify-only
```

Все исходные архивы, фото, веса и индекс находятся в игнорируемом `data/`. Папка `data/models/siglip2-base-patch16-384` занимает около 1,5 GiB. Индекс: `data/index/siglip2.npz` и `data/index/siglip2.json`. `scanner.model_fetch` загружает только закреплённую ревизию модели; `scanner.vision build` отклоняет изменение эталонов относительно манифеста.

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
.venv/bin/python -m scanner.evaluate \
  --images-dir data/official_real_photos \
  --labels evaluation/manual_official_labels.tsv \
  --output data/evaluation/official_predictions.jsonl
```

Рядом создаётся `data/evaluation/official_predictions.summary.json` с точностью на подтверждённых примерах, p50/p95 задержки, версиями ПО, именем GPU, ревизией Git и хешами модели, индекса, манифеста и меток. Метрика удобна для регрессий на этих примерах; для выбора следующей модели нужна отдельная удержанная группа снимков по сеансам съёмки.

## Известные ограничения и следующий шаг

Снимок должен показывать целевую бутылку ближе к центру. Модель иногда путает варианты одной серии или год урожая; 11 эталонных файлов в источнике привязаны сразу к двум `slug`, поэтому изображение не всегда позволяет выбрать точную карточку. Фотографии неизвестного вина и кадры с несколькими бутылками требуют отдельного правила оценки от организаторов. OCR и обучение на синтетических фотографиях пока исследуются и не включены в рабочий маршрут без подтверждённого улучшения на удержанных реальных фото. Подробности — в [ARCHITECTURE.md](ARCHITECTURE.md).
