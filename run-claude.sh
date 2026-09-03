#!/bin/bash

cd /home/vmsam/src/VMSAM

FLAGS=(--dangerously-skip-permissions --remote-control "vmsam-dev-sandbox"
       --add-dir "/srv" "/tmp" "/config" "${folder_to_watch}" "${folder_error}" "/home/vmsam/src")
claude "${FLAGS[@]}" --continue || claude "${FLAGS[@]}"