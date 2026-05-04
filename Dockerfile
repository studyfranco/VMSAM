# ---------- builder ----------
FROM rust:slim AS builder
WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates pkg-config libssl-dev git \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/* \
    && mkdir src

# copy manifest and fetch deps
COPY Cargo.toml ./
RUN echo 'fn main() { println!("dummy"); }' > src/main.rs && cargo build --release

# copy sources
COPY src/lib.rs src/main.rs src/
# build release
ENV RUSTFLAGS="-C opt-level=3 -C strip=symbols"
RUN cargo build --release

# ---------- Runtime ----------
FROM ghcr.io/studyfranco/docker-baseimages-debian:testing-video

LABEL maintainer="studyfranco@gmail.com"

ARG defaultlibvmaf="https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz" \
    pathtomodelfromdownload="vmaf-3.0.0/model"

# && echo "deb https://deb.debian.org/debian/ bullseye main contrib non-free" >> /etc/apt/sources.list.d/bullseye.list \
RUN set -x \
    && apt update \
    && apt dist-upgrade -y \
    && apt autopurge -yy \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

RUN set -x \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y tar gosu libchromaprint-tools mediainfo ffmpeg mkvtoolnix sqlite3 --no-install-recommends \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

RUN set -x \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y python3-numpy python3-scipy python3-matplotlib python3-onnxruntime python3-resampy python3-sqlalchemy python3-sqlalchemy-ext python3-psycopg python3-fastapi python3-uvicorn python3-dotenv python3-pydantic-settings python3-pip python3-psutil python3-pysubs2 --no-install-recommends \
    && python3 -m pip install --break-system-packages iso639-lang \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/* /root/.cache

RUN set -x \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y wget \
    && useradd -ms /bin/bash vmsam \
    && gosu nobody true \
    && mkdir -p /config \
    && chown -R vmsam:vmsam /config \
    && mkdir -p /usr/share/model \
    && mkdir -p /libvmaf \
    && wget --no-check-certificate $defaultlibvmaf -O /libvmaf/libvmaf.tar.gz \
    && tar -xzf /libvmaf/libvmaf.tar.gz -C /libvmaf/ \
    && mv /libvmaf/$pathtomodelfromdownload/* /usr/share/model/ \
    && rm -r /libvmaf \
    && DEBIAN_FRONTEND=noninteractive apt purge -y wget \
    && apt autopurge -yy \
    && apt clean autoclean -y \
    && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

ENV CORE=4 \
    WAIT=300 \
    PGID="1000" \
    PUID="1000" \
    software="main" \
    folder_to_watch="/config/input" \
    folder_error="/config/error" \
    dev=false

RUN mkdir -p /home/vmsam/gestionar_show/ \
    && mkdir -p /home/vmsam/gestionar_movie/
COPY init.sh /
COPY --chown=vmsam:vmsam src/*.ini src/*.py run.sh src/titles_subs_group.json src/config.json /home/vmsam/
COPY --chown=vmsam:vmsam src/gestionar_show /home/vmsam/gestionar_show/
COPY --chown=vmsam:vmsam src/gestionar_movie /home/vmsam/gestionar_movie/
COPY --from=builder --chown=vmsam:vmsam /usr/src/app/target/release/audio_sync /home/vmsam/audio_sync
RUN chmod +x /init.sh \
    && chmod +x /home/vmsam/run.sh

WORKDIR /

ENTRYPOINT [ "/init.sh" ]