from debian:testing-slim

LABEL maintainer="studyfranco@gmail.com"

COPY init.sh /

RUN set -x \
 && apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y gosu libchromaprint-tools ffmpeg mediainfo mkvtoolnix python3 python3-numpy python3-scipy python3-matplotlib --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -ms /bin/bash vmsam \
 && gosu nobody true \
 && mkdir -p /config/input \
 && mkdir -p /config/error \
 && mkdir -p /config/output \
 && chown -R vmsam:vmsam /config \
 && chmod +x /init.sh

COPY --chown=vmsam:vmsam src/*.ini src/*.py /home/vmsam/

WORKDIR /

ENV CORE=4 \
    WAIT=300 \
    PGID="1000" \
    PUID="1000"

ENTRYPOINT [ "/init.sh" ]

