# Эмбеддинги этикеток

`BottleEmbedder` использует актуальный чекпойнт
`embedding/runs/dinov2_large_labels/best.pt` и возвращает L2-нормированный
вектор из 512 чисел. Передавайте кроп этикетки; модель сама этикетку на
полной фотографии не ищет.

```python
from bottle_embeddings import BottleEmbedder, load_database, search_embeddings

embedder = BottleEmbedder(device="cuda")
vector = embedder.embed_image("data/competition_label_crops/crops/example.jpg")
assert vector.shape == (512,)

database = load_database("competition_labels_embeddings.json")
matches = search_embeddings(vector, database, top_k=5)
```

Для получения кропов и базы из файлов `df_2.csv`:

```bash
detection/.venv/bin/python -u -m embedding.competition_labels --stage all --crop-batch 8 --embed-batch 8 --device cuda
```

База находится в `competition_labels_embeddings.json`: `fname -> список
детекций`, у каждой детекции есть `crop_path`, `xyxy`, `crop_xyxy`,
`confidence`, `embedding`. Полный порядок действий описан в
`data/competition_label_crops/README.md`.
