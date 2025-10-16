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
FROM ghcr.io/studyfranco/docker-baseimages-debian:testing

LABEL maintainer="studyfranco@gmail.com"

ARG defaultlibvmaf="https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz" \
    pathtomodelfromdownload="vmaf-3.0.0/model"

# Update and upgrade system packages
RUN set -x \
 && apt update \
 && apt dist-upgrade -y \
 && apt autopurge -yy \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

# Install core multimedia and processing tools
RUN set -x \
 && apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y tar gosu libchromaprint-tools mediainfo ffmpeg mkvtoolnix sqlite3 --no-install-recommends \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

# Install Python packages and ML dependencies for scene detection
RUN set -x \
 && apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y python3-numpy python3-scipy python3-matplotlib python3-onnxruntime python3-resampy python3-sqlalchemy python3-sqlalchemy-ext python3-psycopg python3-fastapi python3-uvicorn python3-dotenv python3-pydantic-settings python3-pip python3-psutil python3-pysubs2 --no-install-recommends \
 && python3 -m pip install --break-system-packages iso639-lang \
 # Install PySceneDetect and additional ML packages for scene detection \
 && python3 -m pip install --break-system-packages scenedetect[opencv] \
 && python3 -m pip install --break-system-packages opencv-python-headless \
 # Install additional packages for enhanced frame comparison and uncertainty analysis \
 && python3 -m pip install --break-system-packages scikit-image \
 && python3 -m pip install --break-system-packages imageio \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/* /root/.cache

# Setup user and download VMAF models
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

# Environment variables with ML scene detection support
ENV CORE=4 \
    WAIT=300 \
    PGID="1000" \
    PUID="1000" \
    software="main" \
    folder_to_watch="/config/input" \
    folder_error="/config/error" \
    dev=false \
    # ML scene detection settings \
    VMSAM_ML_SCENE_DETECTION=true \
    VMSAM_SCENE_THRESHOLD=27.0 \
    VMSAM_MIN_SCENE_LEN=30

# Copy application files including new ML modules
COPY init.sh /
COPY --chown=vmsam:vmsam src/*.ini src/*.py run.sh src/titles_subs_group.json src/config.json /home/vmsam/
COPY --from=builder --chown=vmsam:vmsam /usr/src/app/target/release/audio_sync /home/vmsam/audio_sync
RUN chmod +x /init.sh \
 && chmod +x /home/vmsam/run.sh

WORKDIR /

ENTRYPOINT [ "/init.sh" ]