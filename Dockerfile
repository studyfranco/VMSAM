FROM ghcr.io/studyfranco/docker-baseimages-debian:testing

LABEL maintainer="studyfranco@gmail.com"

ARG defaultlibvmaf="https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz" \
    pathtomodelfromdownload="vmaf-3.0.0/model"

RUN set -x \
 && echo "deb https://deb.debian.org/debian/ bullseye main contrib non-free" >> /etc/apt/sources.list.d/bullseye.list \
 && apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y tar wget gosu libchromaprint-tools=1.5.0-2 mediainfo ffmpeg mkvtoolnix python3 python3-numpy python3-scipy python3-matplotlib --no-install-recommends \
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
 && apt dist-upgrade -y \
 && apt autopurge -yy \
 && apt clean autoclean -y \
 && rm -rf /var/cache/* /var/lib/apt/lists/* /var/log/* /var/tmp/* /tmp/*

COPY init.sh /
COPY --chown=vmsam:vmsam src/*.ini src/*.py run.sh /home/vmsam/
RUN chmod +x /init.sh \
 && chmod +x /home/vmsam/run.sh

WORKDIR /

ENV CORE=4 \
    WAIT=300 \
    PGID="1000" \
    PUID="1000"

ENTRYPOINT [ "/init.sh" ]