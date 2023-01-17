from debian:testing-slim

LABEL maintainer="studyfranco@gmail.com"

RUN set -x \
 && echo "deb http://www.deb-multimedia.org testing main non-free" >> /etc/apt/sources.list.d/multimedia.list \
 && apt update -oAcquire::AllowInsecureRepositories=true \
 && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated deb-multimedia-keyring gosu libchromaprint-tools mediainfo ffmpeg mkvtoolnix python3 python3-numpy python3-scipy python3-matplotlib --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -ms /bin/bash vmsam \
 && gosu nobody true \
 && mkdir -p /config/input \
 && mkdir -p /config/error \
 && mkdir -p /config/output \
 && chown -R vmsam:vmsam /config

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
