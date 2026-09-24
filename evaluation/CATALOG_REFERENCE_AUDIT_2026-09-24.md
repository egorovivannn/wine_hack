# Аудит эталонов каталога, 24 сентября 2026

## Результат

Исходный `df_2.csv` и его проверенный манифест не изменены. Отдельный исправленный манифест меняет **три** связи `slug -> изображение` после сверки названия товара с этикеткой и файлом исходного Strapi-архива. Оба новых файла извлекаются из уже переданного организаторами архива; третье изображение уже используется в каталоге. Все байты закреплены SHA-256 в [`manifests/catalog_reference_fixes_2026-09-24.json`](manifests/catalog_reference_fixes_2026-09-24.json).

| Карточка | Что было | Подтверждённый эталон | Основание |
| --- | --- | --- | --- |
| `vibes-vermentino-viognier-barrel-fermented-2022` | Бутылка Silvaner | `vibes_vermentino_viognier_barrel_fermented_vermentino_beloe_suhoe_115_fe273c3a1b.webp` | Исходный файл и этикетка относятся к Silvaner; новый файл явно относится к Vermentino–Viognier Barrel Fermented. |
| `sand-zh-oveze-rezerv` | Этикетка Rebo Reserve | `Sandzhoveze_reserve_2023_9c8ad3555e.webp` | На новой этикетке написано «Санджовезе резерв». |
| `vinodelnya-byurne-byurne-pino-blan-beloe-suhoe-14` | Этикетка «Вионье» | `vinodelnya_byurne_byurne_pino_blan_beloe_suhoe_14_45558bffe3.webp` | На новой этикетке написано «Пино Блан». |

В исходном каталоге 2 103 карточки, 2 098 с эталоном, 2 087 разных файлов и 11 файлов, общих для двух `slug`. В исправленном варианте: 2 089 разных файлов и 9 общих. Это исправление **данных**, а не измеренный прирост точности; для российского полочного домена независимой размеченной проверки пока нет.

## Неисправленные случаи

Пять карточек с `is_valid_img=False` не имеют подтверждённого точного изображения в переданном источнике: `fanagoriya-fanagoriya-hey-bey-shardone-beloe-suhoe-13`, `katharon-mezenka-risling-polusuhoe`, `katharon-mezenka-risling-suhoe`, `leto-kaberne-fran-2021-suhoe-krasnoe`, `vinodelnya-vedernikov-vedernikov-tsimlyanskiy-chyornyy-tsimlyanskiy-chernyy-krasnoe-suhoe-14`. Похожие по словам изображения относятся к другим винам; подставлять их нельзя.

Все 11 исходных общих эталонов просмотрены по изображению и полям обеих карточек:

| Общий файл | Вывод |
| --- | --- |
| `03_Silvaner_2022_Barrel_Fermented_01addc1e7f.webp` | Исправлен Vermentino–Viognier. |
| `4285_eq_F1_Fau_no_bg_preview_carve_photos_6d7da0f321.webp` | Две карточки «Алиготе Авторское» с одной этикеткой; отдельный товар не подтверждён. |
| `4300_tl_K_Hp_Eg_0fa0eef9b4.webp` | Этикетка «Новый Свет полусладкое» не подтверждает карточку брют; точного отдельного эталона для неё не установлено. |
| `Agora_Rezerv_Yahting_Sovinon_kopiya_066436e3ad.webp` | На фото красное Agora Sauvignon, тогда как одна карточка заявлена белым Совиньон Блан; альтернативный точный эталон не установлен. |
| `Aligote_Czitron_bel_suh_1cc2763d35.webp` | На фото Алиготе–Цитрон, а вторая карточка — Цитрон–Шардоне. В архиве есть `Czitron_Shard_p_sl_b94274915f.webp`, но он обозначен как полусладкий и не подтверждает сухую версию второй карточки. |
| `DSC_00836_4070f8fd2f.webp` | На фото Пино Нуар 2025, не Method Classic Кокур; точный второй эталон не найден. |
| `DSC_00839_Photoroom_418c4b31b3.webp` | Две карточки Кокур 2025 с одной этикеткой; отдельный товар не подтверждён. |
| `DSC_09173_4a9ff95cc2.webp` | Два года Алиготе Баррель; на фото год достоверно не различается. |
| `Rebo_reserve_2023_2f54a334ff.webp` | Исправлен Санджовезе резерв. |
| `Vione_2023_13d6c1ebf6.webp` | Исправлен Пино Блан. |
| `bottle_02_Cabernet_Franc_ec42fb71c4.webp` | Два года Cabernet Franc–Pinot Noir; на фото год достоверно не различается. |

Неверные связи без точного заменяющего изображения оставлены в исходном манифесте и перечислены выше как ограничение. Это сохраняет исходные данные и позволяет повторить сравнение. В частности, общий файл сам по себе не доказывает, какой из двух `slug` требуется в скрытом тесте.

## Воспроизведение

```bash
uv run --locked python -m scanner.prepare --verify-only
uv run --locked python -m scanner.reference_fixes
uv run --locked python -m scanner.vision build \
  --manifest data/catalog_manifest_corrected.json \
  --index data/index/siglip2_corrected.npz --batch-size 16
WINE_MANIFEST=data/catalog_manifest_corrected.json \
WINE_INDEX=data/index/siglip2_corrected.npz \
uv run --locked uvicorn scanner.server:app --host 127.0.0.1 --port 8088
```

`scanner.reference_fixes` проверяет SHA исходного манифеста и каждого целевого файла, извлекает отсутствующие файлы из архива и пишет **отдельный** манифест. Исходный CSV, архив и индекс остаются неизменными. Размеры и хеши исправленного индекса записываются рядом с ним в `data/index/siglip2_corrected.json`. Официальные 100 фото и собственные 13 магазинных кадров для этого аудита не использовались.
