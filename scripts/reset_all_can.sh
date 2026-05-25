#!/bin/bash
# Deprecated wrapper — use scripts/setup_devices.sh
exec "$(dirname "$0")/setup_devices.sh" "$@"
