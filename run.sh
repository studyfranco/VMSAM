#!/bin/bash

if [[ "${software}" == "main" ]]; then
    python3 /home/vmsam/infinite_wrapper.py --pwd /home/vmsam --folder /config/input -o /config/output -e /config/error -c $CORE -w $WAIT
else if [[ "${software}" == "gestionar_show" ]]; then
    python3 /home/vmsam/gestionar_show.py --pwd /home/vmsam --folder /config/input -e /config/error -c $CORE -w $WAIT
else
    echo "Unknown software option: ${software}"
    exit 1
fi
exit 0