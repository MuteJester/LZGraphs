# LZGraphs container image, published to GHCR so the package shows up in
# the repository's Packages sidebar. Runs the `lzg` CLI as its entrypoint.
#
# Two stages:
#   1. builder: a full python:slim image plus build-essential (gcc, make,
#      ...), used to compile LZGraphs' C extension into a wheel. LZGraphs
#      has no prebuilt manylinux/musllinux wheel for every possible base
#      image, and this image needs to build from source anyway, so a base
#      that can run a real C compiler is required here, not optional.
#   2. final: the same slim base, with only the built wheel (and the numpy
#      dependency it pulls in) installed. No compiler, no source tree, no
#      pip cache. This is ordinary multi-stage hygiene, kept because it is
#      close to free, not an attempt at an aggressively minimal image.
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN python -m pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel -o /dist

FROM python:3.12-slim

COPY --from=builder /dist/*.whl /tmp/wheels/
RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

ENTRYPOINT ["lzg"]
CMD ["--help"]
