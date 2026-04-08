FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip python3-venv \
    ffmpeg libsndfile1 \
    >/dev/null 2>&1

COPY pyproject.toml /app/
COPY src/ /app/src/

WORKDIR /app

RUN pip install --no-cache-dir --break-system-packages -e ".[kokoro]"

ENTRYPOINT ["unkork"]
