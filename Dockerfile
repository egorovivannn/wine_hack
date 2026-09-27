# Runtime image for the scanner API. Data, weights and the index are mounted from ./data;
# prepare them on the host with the commands in README.md.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --python /usr/bin/python3.12
COPY scanner ./scanner
COPY df_2.csv ./

EXPOSE 8088
CMD ["uv", "run", "--locked", "--no-sync", "uvicorn", "scanner.server:app", "--host", "0.0.0.0", "--port", "8088", "--workers", "1"]
