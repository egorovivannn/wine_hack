# Scanner image. Mount ./data with the organizers' archive at data/source/Датасет.zip: on the first start
# the container prepares the catalogue, downloads the pinned models and builds the index there, then serves.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=4 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates libgl1 libglib2.0-0 7zip 7zip-rar \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# The CUDA wheels for torch are several GB; keep finished downloads in a cache and retry on network errors.
RUN --mount=type=cache,target=/root/.cache/uv \
    for attempt in 1 2 3; do \
      uv sync --locked --no-install-project --python /usr/bin/python3.12 && exit 0; \
      echo "uv sync failed (attempt $attempt), retrying"; sleep 10; \
    done; exit 1
COPY scanner ./scanner
COPY df_2.csv ./
COPY evaluation/manifests ./evaluation/manifests
COPY docker/entrypoint.sh /usr/local/bin/wine-entrypoint

EXPOSE 8088
CMD ["wine-entrypoint"]
