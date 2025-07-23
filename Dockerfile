FROM ghcr.io/studyfranco/docker-baseimages-debian:testing

LABEL maintainer="studyfranco@gmail.com"

ARG defaultlibvmaf="https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz" \
    pathtomodelfromdownload="vmaf-3.0.0/model"

RUN set -x \
 && echo "deb https://deb.debian.org/debian/ bullseye main contrib non-free" >> /etc/apt/sources.list.d/bullseye.list \
 && apt update \
 && apt dist-upgrade -y \
 && apt autopurge -yy \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

RUN set -x \
 && apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y tar gosu libchromaprint-tools=1.5.0-2 mediainfo ffmpeg mkvtoolnix sqlite3 --no-install-recommends \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

RUN set -x \
 && apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-numpy python3-scipy python3-matplotlib python3-onnxruntime python3-resampy python3-sqlalchemy python3-sqlalchemy-ext python3-psycopg python3-fastapi python3-uvicorn python3-dotenv python3-pydantic-settings --no-install-recommends \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

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

COPY init.sh /
COPY --chown=vmsam:vmsam src/*.ini src/*.py run.sh src/titles_subs_group.json /home/vmsam/
RUN chmod +x /init.sh \
 && chmod +x /home/vmsam/run.sh

WORKDIR /

ENV CORE=4 \
    WAIT=300 \
    PGID="1000" \
    PUID="1000" \
    software="main" \
    folder_to_watch="/config/input" \
    folder_error="/config/error" 

ENTRYPOINT [ "/init.sh" ]