# syntax=docker/dockerfile:1.4
FROM ghcr.io/vantage6/infrastructure/algorithm-base:4.15

# System dependencies + CUDA keyring in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ffmpeg libsm6 libxext6 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python --version

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_GPU_ALLOCATOR=cuda_malloc_async   

WORKDIR /app

# Upgrade pip/setuptools first (cached unless this layer is invalidated)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools

# Copy only dependency manifest first, so dependency layer is cached
# independently of source code changes
COPY pyproject.toml /app/pyproject.toml

# Install dependencies only (no source yet, so this can't pull in your package).
# --no-deps would fail here since there's no source; instead we install with a
# dummy minimal build so pip resolves and caches all the heavy deps.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip tomli && \
    python -c "import tomli; d=tomli.load(open('pyproject.toml','rb')); print('\n'.join(d['project']['dependencies']))" > requirements.txt && \
    pip install -r requirements.txt

# Now copy the actual source code (this layer changes often, but everything
# above it stays cached)
COPY . /app

ARG PKG_NAME="argosfeddeep"

# Install your own package without re-resolving/re-downloading dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /app

ENV PKG_NAME=${PKG_NAME}
ENV TF_USE_LEGACY_KERAS=1
ENV TF_DISABLE_MLIR_CRASH_REPRODUCER=1
ENV CUDA_MODULE_LOADING=LAZY


CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"