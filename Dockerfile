from debian:testing-slim

LABEL maintainer="studyfranco@gmail.com"

ARG defaultlibvmaf="https://github.com/Netflix/vmaf/archive/refs/tags/v2.3.1.tar.gz" \
    pathtomodelfromdownload="vmaf-2.3.1/model"

RUN set -x \
 && echo "deb http://www.deb-multimedia.org testing main non-free" >> /etc/apt/sources.list.d/multimedia.list \
 && echo "deb http://deb.debian.org/debian/ stable main contrib non-free" >> /etc/apt/sources.list.d/stable.list \
 && apt update -oAcquire::AllowInsecureRepositories=true \
 && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated deb-multimedia-keyring tar wget gosu libchromaprint-tools=1.5.0-2 mediainfo ffmpeg mkvtoolnix python3 python3-numpy python3-scipy python3-matplotlib --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -ms /bin/bash vmsam \
 && gosu nobody true \
 && mkdir -p /config/input \
 && mkdir -p /config/error \
 && mkdir -p /config/output \
 && mkdir -p /config/models \
 && chown -R vmsam:vmsam /config \
 && mkdir -p /usr/share/model \
 && mkdir -p /libvmaf \
 && wget --no-check-certificate $defaultlibvmaf -O /libvmaf/libvmaf.tar.gz \
 && tar -xzf /libvmaf/libvmaf.tar.gz -C /libvmaf/ \
 && mv /libvmaf/$pathtomodelfromdownload/* /usr/share/model/ \
 && rm -r /libvmaf

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
