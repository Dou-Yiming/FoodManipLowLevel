#!/usr/bin/env bash
# One-time install: copy setup script + udev rule so CAN comes up on plug-in.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root: sudo bash scripts/install_hotplug_setup.sh"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SETUP_SRC="$REPO_ROOT/scripts/setup_devices.sh"
SETUP_DST="/usr/local/bin/foodmanip-setup-devices"
UDEV_SRC="$REPO_ROOT/udev/99-foodmanip-hotplug.rules"
UDEV_DST="/etc/udev/rules.d/99-foodmanip-hotplug.rules"

if [[ ! -f "$SETUP_SRC" ]]; then
    echo "Missing $SETUP_SRC"
    exit 1
fi

echo "[install] Installing $SETUP_DST"
install -m 0755 "$SETUP_SRC" "$SETUP_DST"

echo "[install] Installing udev rule -> $UDEV_DST"
install -m 0644 "$UDEV_SRC" "$UDEV_DST"

# Optional: CAN rename rules + user groups from repo helpers
if [[ -f "$REPO_ROOT/udev/80-i2rt-can.rules" ]]; then
    echo "[install] Installing 80-i2rt-can.rules"
    install -m 0644 "$REPO_ROOT/udev/80-i2rt-can.rules" /etc/udev/rules.d/80-i2rt-can.rules
fi

if [[ -x "$REPO_ROOT/devices/install_devices.sh" ]]; then
    echo "[install] Running devices/install_devices.sh (groups + optional rules)"
    sh "$REPO_ROOT/devices/install_devices.sh"
fi

echo "[install] Reloading udev..."
udevadm control --reload-rules
udevadm trigger --subsystem-match=net --action=add

echo "[install] Running setup once for currently connected devices..."
"$SETUP_DST"

echo "[install] Hotplug setup complete."
echo "  Manual:  sudo foodmanip-setup-devices"
echo "  Or:      sudo bash scripts/setup_devices.sh"
