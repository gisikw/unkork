FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip python3-venv \
    ffmpeg libsndfile1 \
    >/dev/null 2>&1

WORKDIR /app

# Allow pip installs system-wide (PEP 668) — both at build time and
# runtime, since Kokoro's KPipeline pip-installs dependencies lazily.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install deps from pyproject.toml without building the local package.
# Source is copied after so code changes don't bust the pip layer cache.
COPY pyproject.toml /app/
RUN pip install --no-cache-dir \
    torch numpy scipy resemblyzer cma soundfile click \
    kokoro kokoro-onnx

COPY src/ /app/src/
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["unkork"]
