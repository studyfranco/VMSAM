# Build stage
FROM rust:slim as builder

WORKDIR /usr/src/app
COPY . .

RUN set -x \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y build-essential ca-certificates pkg-config libssl-dev git --no-install-recommends \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

RUN cargo build --release

# Runtime stage
FROM ghcr.io/studyfranco/docker-baseimages-debian:testing

RUN set -x \
    && apt update \
    && apt dist-upgrade -y \
    && apt autopurge -yy \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

RUN set -x \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y ca-certificates --no-install-recommends \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

WORKDIR /app
COPY --from=builder /usr/src/app/target/release/vmsam-web /app/vmsam-web
COPY static /app/static

CMD ["./vmsam-web"]
