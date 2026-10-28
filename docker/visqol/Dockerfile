# ViSQOL builder, built once and published under a fixed tag: the app image copies /visqol/ from it.
FROM debian:bookworm AS build

ARG VISQOL_COMMIT=38d0b0163e441047d4429bf07ad09e5b9031d02c \
    BAZELISK_VERSION=v1.25.0

RUN set -x \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential ca-certificates curl git patch python3 python3-dev python3-distutils python3-numpy unzip zip \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHON_BIN_PATH=/usr/bin/python3

RUN set -x \
    && curl -sSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

WORKDIR /src
RUN set -x \
    && git clone https://github.com/google/visqol.git . \
    && git checkout ${VISQOL_COMMIT} \
    && bazel build :visqol -c opt \
    && mkdir -p /visqol \
    && cp -L bazel-bin/visqol /visqol/visqol \
    && strip /visqol/visqol \
    && cp model/libsvm_nu_svr_model.txt LICENSE /visqol/

FROM scratch
COPY --from=build /visqol/ /visqol/