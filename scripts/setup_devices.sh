#!/usr/bin/env bash
# Bring up CAN interfaces and refresh device state after plug/unplug.
# Run manually after connecting hardware, or install hotplug via install_hotplug_setup.sh.
set -euo pipefail

BITRATE="${CAN_BITRATE:-1000000}"
DELAY_SEC="${SETUP_DEVICES_DELAY:-0.5}"
LOG_TAG="foodmanip-setup"
LOCK_FILE="/run/foodmanip-setup.lock"

if [[ "$(id -u)" -ne 0 ]]; then
    SUDO="sudo"
else
    SUDO=""
fi

log() {
    if command -v logger >/dev/null 2>&1; then
        logger -t "$LOG_TAG" "$*"
    fi
    echo "[setup_devices] $*"
}

list_can_interfaces() {
    ip -o link show 2>/dev/null | awk -F': ' '/: can/ {print $2}' | cut -d@ -f1 | sort -u
}

bring_up_can() {
    local iface="$1"
    log "CAN $iface -> down, up @ ${BITRATE} bps"
    $SUDO ip link set "$iface" down 2>/dev/null || true
    if ! $SUDO ip link set "$iface" up type can bitrate "$BITRATE"; then
        log "WARN: failed to bring up $iface"
        return 1
    fi
    return 0
}

main() {
    # Debounce: udev may fire once per adapter in quick succession.
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "Another setup run in progress; skipping."
        exit 0
    fi

    sleep "$DELAY_SEC"

    mapfile -t can_ifaces < <(list_can_interfaces)
    if [[ ${#can_ifaces[@]} -eq 0 ]]; then
        log "No CAN interfaces found (plug in CAN adapters and retry)."
    else
        log "Found CAN: ${can_ifaces[*]}"
        for iface in "${can_ifaces[@]}"; do
            bring_up_can "$iface" || true
        done
    fi

    if [[ -d /dev ]]; then
        mapfile -t video_devs < <(ls /dev/video* 2>/dev/null | sort -V || true)
        if [[ ${#video_devs[@]} -gt 0 ]]; then
            log "Video devices: ${video_devs[*]}"
        fi
    fi

    log "Done."
}

main "$@"
