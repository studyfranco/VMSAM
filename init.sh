#!/bin/bash

set -e

CURRENTUID=$(id -u)
NUMCHECK='^[0-9]+$'
USER="vmsam"

if [[ "$CURRENTUID" -ne "0" ]]; then
    printf "Current user is not root (%s)\\nPass your user and group to the container using the PGID and PUID environment variables\\nDo not use the --user flag (or user: field in Docker Compose)\\n" "$CURRENTUID"
    exit 1
fi

# check if the user and group IDs have been set
if ! [[ "$PGID" =~ $NUMCHECK ]] ; then
    printf "Invalid group id given: %s\\n" "$PGID"
    PGID="1000"
elif [[ "$PGID" -eq 0 ]]; then
    printf "PGID/group cannot be 0 (root)\\n"
    exit 1
fi

if ! [[ "$PUID" =~ $NUMCHECK ]] ; then
    printf "Invalid user id given: %s\\n" "$PUID"
    PUID="1000"
elif [[ "$PUID" -eq 0 ]]; then
    printf "PUID/user cannot be 0 (root)\\n"
    exit 1
fi

if [[ $(getent group $PGID | cut -d: -f1) ]]; then
    usermod -a -G "$PGID" "$USER"
else
    groupmod -g "$PGID" "$USER"
fi

if [[ $(getent passwd ${PUID} | cut -d: -f1) ]]; then
    USER=$(getent passwd $PUID | cut -d: -f1)
else
    usermod -u "$PUID" "$USER"
fi

mkdir -p /config/input
mkdir -p /config/error
mkdir -p /config/output
mkdir -p /config/models

chown -R "$PUID":"$PGID" /config /home/vmsam
rm -r /tmp/* || true
chown -R "$PUID":"$PGID" /tmp
exec gosu "$USER" "/home/vmsam/run.sh"
