# uvcvideo-mjpg — DKMS module for multi-camera USB rigs

The stock Linux UVC driver reserves the USB bandwidth each camera's firmware
*advertises* — an inflated worst-case that caps a 480 Mbps USB-2 bus at two
concurrent cameras, even for lightweight MJPEG streams (the third fails with
`VIDIOC_STREAMON: No space left on device`). The kernel's `FIX_BANDWIDTH`
quirk computes realistic reservations but is gated to uncompressed formats.
This module removes that gate — one changed condition in
`uvc_video.c:uvc_fixup_video_ctrl()`, marked `MakerLab patch` — letting
MJPEG streams reserve realistically (~65 Mbps each → 5+ cameras per bus).
The quirk must be enabled (`quirks=128`) for the patch to do anything; the
installer sets that via `/etc/modprobe.d/uvcvideo.conf`.

Install: `sudo ./install.sh` (details + verification in ../../JETSON_SETUP.md).
DKMS rebuilds the module automatically on kernel updates.

`src/` is `drivers/media/usb/uvc/` from mainline Linux **v6.8.12** (matching
the JetPack 7 R39 kernel) plus the one-line patch. These files are
**GPL-2.0** (kernel code, SPDX headers intact) — they are a standalone kernel
module, independent of the application code in this repository. To move to a
new kernel series, refresh `src/` from the matching mainline tag and re-apply
the patch (grep for `MakerLab patch`).
