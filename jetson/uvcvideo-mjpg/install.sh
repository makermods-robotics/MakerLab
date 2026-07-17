#!/usr/bin/env bash
# Install the uvcvideo-mjpg DKMS module: the stock uvcvideo driver with the
# FIX_BANDWIDTH quirk extended to compressed (MJPEG) formats, so more than
# two USB cameras can stream concurrently on one USB-2 bus. Background and
# verification: jetson/README.md, "Multi-camera USB".
#
# Usage: sudo ./install.sh
# Rollback: sudo dkms remove uvcvideo-mjpg/6.8.12 --all
#           sudo rm /etc/modprobe.d/uvcvideo.conf && sudo depmod -a
set -euo pipefail

NAME=uvcvideo-mjpg
VERSION=6.8.12   # kernel version the vendored src/ was taken from (mainline)
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/src"
DEST="/usr/src/${NAME}-${VERSION}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

KREL="$(uname -r)"
case "$KREL" in
  6.8.*) ;;
  *)
    echo "WARNING: vendored source is from mainline v${VERSION}; you are on ${KREL}." >&2
    echo "The build may fail or misbehave — refresh src/ from your kernel version first." >&2
    read -rp "Continue anyway? [y/N] " ans
    [ "${ans,,}" = "y" ] || exit 1
    ;;
esac

apt-get install -y dkms "linux-headers-${KREL}"

# Stage the source where DKMS expects it (fresh copy each run)
rm -rf "$DEST"
mkdir -p "$DEST"
cp "$SRC_DIR"/*.c "$SRC_DIR"/*.h "$SRC_DIR"/Makefile "$DEST"/
cat > "$DEST/dkms.conf" <<EOF
PACKAGE_NAME="$NAME"
PACKAGE_VERSION="$VERSION"
BUILT_MODULE_NAME[0]="uvcvideo"
DEST_MODULE_LOCATION[0]="/updates"
AUTOINSTALL="yes"
MAKE[0]="make -C \${kernel_source_dir} M=\${dkms_tree}/\${PACKAGE_NAME}/\${PACKAGE_VERSION}/build modules"
CLEAN="make -C \${kernel_source_dir} M=\${dkms_tree}/\${PACKAGE_NAME}/\${PACKAGE_VERSION}/build clean"
EOF

# Re-register idempotently, then build + install for the running kernel
if dkms status -m "$NAME" -v "$VERSION" 2>/dev/null | grep -q "$NAME"; then
  dkms remove "$NAME/$VERSION" --all
fi
dkms install "$NAME/$VERSION"

# The patch only activates with the FIX_BANDWIDTH quirk set
echo 'options uvcvideo quirks=128' > /etc/modprobe.d/uvcvideo.conf

# Swap the running module unless a camera is currently streaming
if fuser -s /dev/video* 2>/dev/null; then
  echo "A camera is in use — close camera apps, then run:" >&2
  echo "  sudo rmmod uvcvideo && sudo modprobe uvcvideo   (or reboot)" >&2
else
  rmmod uvcvideo 2>/dev/null || true
  modprobe uvcvideo
fi

echo
echo "module : $(modinfo -n uvcvideo)"   # expect .../updates/dkms/uvcvideo.ko
echo "quirks : $(cat /sys/module/uvcvideo/parameters/quirks 2>/dev/null || echo '(module not loaded)')"
echo "Done. Verify with three concurrent MJPG streams once cameras are idle (jetson/README.md)."
