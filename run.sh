#!/bin/bash

if [[ "${software}" == "main" ]]; then
    python3 -u /home/vmsam/infinite_wrapper.py --pwd /home/vmsam --folder /config/input -o /config/output -e /config/error -c $CORE -w $WAIT
elif [[ "${software}" == "gestionar_show" ]]; then
    python3 -u /home/vmsam/main_gestionar_show.py --pwd /home/vmsam --folder "${folder_to_watch}" -e ${folder_error} -c $CORE -w $WAIT --database_url_file /database_url.json
else
    echo "Unknown software option: ${software}"
    exit 1
fi
exit 0