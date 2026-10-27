# ViSQOL builder: compiles the ViSQOL command line tool once; the app image copies only /visqol/ from it.
ARG BASE_IMAGE=ghcr.io/studyfranco/docker-baseimages-debian:testing-video

FROM ${BASE_IMAGE} AS build

ARG VISQOL_COMMIT=38d0b0163e441047d4429bf07ad09e5b9031d02c \
    BAZELISK_VERSION=v1.25.0

RUN set -x \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y build-essential ca-certificates curl git patch python3 unzip zip --no-install-recommends \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/*

WORKDIR /src

# GCC 15+ refuses the pinned flatbuffers and TensorFlow Lite sources without these flags (-Wtemplate-body errors, missing <cstdint>).
RUN set -x \
    && curl -sSL -o /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazelisk \
    && curl -sSL https://github.com/google/visqol/archive/${VISQOL_COMMIT}.tar.gz | tar -xz --strip-components=1 \
    && bazelisk build :visqol -c opt \
        --copt=-Wno-template-body --host_copt=-Wno-template-body \
        --cxxopt=-fpermissive --host_cxxopt=-fpermissive \
        --copt=-Wno-error --host_copt=-Wno-error \
        --cxxopt=-include --cxxopt=cstdint --host_cxxopt=-include --host_cxxopt=cstdint \
    && mkdir -p /visqol \
    && cp -L bazel-bin/visqol /visqol/visqol \
    && chmod u+w /visqol/visqol && strip /visqol/visqol \
    && cp model/libsvm_nu_svr_model.txt /visqol/ \
    && cp LICENSE /visqol/LICENSE \
    && test -x /visqol/visqol \
    && bazelisk shutdown \
    && rm -rf /root/.cache /src/*

FROM scratch
COPY --from=build /visqol/ /visqol/
