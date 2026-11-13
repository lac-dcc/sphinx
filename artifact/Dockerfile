# Build from the repository root:
#   docker build -t sphinx-artifact -f artifact/Dockerfile .
# Run (add --gpus all to use a GPU; mount the downloaded archives):
#   docker run --rm -it --gpus all --shm-size=8g -v /path/to/archives:/archives sphinx-artifact
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sphinx
COPY artifact/requirements.txt artifact/requirements.txt
RUN pip install --no-cache-dir -r artifact/requirements.txt

COPY . .

ENV SPHINX_RESULTS=/sphinx/results \
    SPHINX_DATA=/sphinx/data \
    SPHINX_OUTPUT=/sphinx/reproduced

WORKDIR /sphinx/ml_experiments/programl_mlir_performance
CMD ["bash"]
