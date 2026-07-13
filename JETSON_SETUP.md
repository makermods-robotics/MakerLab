# Running LeLab on an NVIDIA Jetson

Notes from bringing LeLab up on a Jetson used as a self-contained robot
station: arms, cameras, display, keyboard, and mouse all plugged into the
Jetson, with the UI in a browser on the Jetson itself (snap Chromium has no
WebGL; the 3D viewer falls back to a placeholder automatically). Headless
operation — UI opened from another machine's browser over the LAN — is also
supported (see "Serving the UI over the LAN"). Everything below was hit in
practice; the happy path hides none of it.

## TL;DR checklist

```bash
# --- one-time system prep ---
sudo usermod -aG dialout $USER          # serial port access (re-login required!)
sudo systemctl disable --now ModemManager   # stop it probing the arms as modems
sudo apt install -y git ffmpeg

# --- install ---
curl -LsSf https://astral.sh/uv/install.sh | sh     # then: source ~/.bashrc
git clone <this-repo> && cd <repo>
uv venv --python 3.12
uv pip install "torch==2.10.0" "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu130   # BEFORE -e . (see "GPU torch")
uv pip install -e .

# --- after install: two Jetson-specific fixes (details in their sections) ---
# 1. point torch at the system cuBLAS pair          (see "cuBLAS")
# 2. patch uvcvideo for >2 concurrent USB cameras   (see "Multi-camera USB")

# --- run ---
.venv/bin/makerlab                  # binds localhost + opens the browser on the Jetson
# (headless alternative: .venv/bin/makerlab --lan, browse http://<jetson-ip>:8000)
```

## JetPack & GPU torch

**GPU work (training / on-device inference) requires JetPack 7.2+** (Ubuntu
24.04, system Python 3.12, CUDA 13). Check your generation with
`head -1 /etc/nv_tegra_release` (R36 = JetPack 6, R38/R39 = JetPack 7). On
**JetPack 6** there is no clean GPU lane (JP6 GPU wheels are cp310-only;
LeLab needs Python ≥ 3.12) — a JP6 box still works as a **CPU-only
teleop/recording/calibration station** with the default PyPI resolution.

The TL;DR's torch line is deliberate on all three counts — **exactly
`torch==2.10.0` + `torchvision==0.25.0`, together, before
`uv pip install -e .`**: the cu130 index already serves torch 2.11–2.13,
which is outside lerobot's `torch>=2.7,<2.11` window, so an unpinned or
torch-only install ends with the resolver replacing CUDA torch with CPU
wheels from PyPI. The pinned pair satisfies lerobot's ranges, so it survives
the `-e .` resolve. (These wheels ship no native `sm_87`, but sm_80 kernels
run fine on Orin — a "no kernel image" error means some *other* package's
kernels, not torch.)

**Verify after the full install.** The snippet deliberately uses `nn.Linear`
rather than a plain matmul: only the former routes through cublasLt, the
failure mode right below — a bare `a @ b` passes even when that is broken.

```bash
python - <<'EOF'
import torch, torchvision, torch.nn as nn
print(torch.__version__, torchvision.__version__)   # expect ...+cu130 on BOTH
print(torch.cuda.is_available())                    # expect True
print(nn.Linear(64, 64).cuda()(torch.randn(8, 64, device="cuda")).sum())
EOF
```

### cuBLAS: symlink the venv pair to the system pair

If training/inference dies at startup with `CUBLAS_STATUS_NOT_INITIALIZED
when calling cublasLtMatmulAlgoGetHeuristic(...)` (at any batch size — so
not OOM): the common Jetson profile export
`LD_LIBRARY_PATH=/usr/local/cuda/lib64` makes the loader mix the venv's
pip-bundled `libcublas` with the *system* `libcublasLt`, and the mismatched
pair fails at init — point both at the system pair:

```bash
cd .venv/lib/python3.12/site-packages/nvidia/cu13/lib
for f in libcublas.so.13 libcublasLt.so.13; do
  [ -f "$f" ] && mv "$f" "$f.pip-bak"
  ln -sf /usr/local/cuda/lib64/$f $f
done
```

Re-run the `nn.Linear` verify above. **Reinstalling torch restores the pip
libraries and re-breaks it** — redo the symlinks. A `CUDNN_STATUS_*` error is the same disease in
`nvidia/cudnn/lib`: same treatment.

### Flashing JetPack 7.2 (no host PC needed)

JetPack 7.2 ships a bootable **Jetson ISO** — write it to a ≥16 GB USB
stick (balenaEtcher; SD-card images are discontinued in 7.2), boot the
Jetson from the stick, install onto the
NVMe. Works on the reference devkits (Orin AGX/NX/Nano Super) as long as
the installed firmware is JetPack-6 generation (BSP ≥ r35.5) — check before
starting, and update QSPI firmware from your current JetPack first if it's
older. It's a full disk wipe: evacuate `~/.cache/huggingface/lerobot/`
(calibrations + datasets), imported checkpoints, and `~/.bashrc` beforehand.

## Multi-camera USB: patch uvcvideo (DKMS)

**Symptom:** the third concurrent USB camera fails —
`VIDIOC_STREAMON returned -1 (No space left on device)` (= USB isochronous
bandwidth, **not disk**), or `NotReadableError: could not start video source`
in the browser — while any two work, regardless of ports/hubs.

**Cause:** the UVC driver reserves the bandwidth the camera's firmware
*advertises* — an inflated worst-case (~200 Mbps even for a MJPG stream using
~10 Mbps), so two reservations fill a 480 Mbps bus. The `quirks=128`
(`FIX_BANDWIDTH`) quirk computes realistic reservations but stock kernels
apply it to uncompressed formats only — useless for MJPEG. One-line patch in
`drivers/media/usb/uvc/uvc_video.c`, `uvc_fixup_video_ctrl()`, removes the
gate (MJPEG then reserves the 1024-byte/µframe floor ≈ 65 Mbps → 5+ cameras
per bus; without `quirks=128` the module stays bone-stock):

```diff
-	if (!(format->flags & UVC_FMT_FLAG_COMPRESSED) &&
-	    stream->dev->quirks & UVC_QUIRK_FIX_BANDWIDTH &&
+	if (stream->dev->quirks & UVC_QUIRK_FIX_BANDWIDTH &&
 	    stream->intf->num_altsetting > 1) {
```

The patched source (mainline v6.8.12 + the diff above) is vendored in this
repo with a self-contained installer:

```bash
sudo jetson/uvcvideo-mjpg/install.sh
```

It installs dkms + headers, registers/builds/installs the module, enables
`quirks=128` via `/etc/modprobe.d/uvcvideo.conf`, and reloads the driver
(deferred with instructions if a camera is busy). Verify afterwards — three
concurrent MJPG streams, where stock fails the third with ENOSPC:

```bash
for d in 0 2 4; do
  v4l2-ctl -d /dev/video$d --set-fmt-video=width=640,height=480,pixelformat=MJPG \
    --stream-mmap --stream-count=100 --stream-to=/dev/null &
done; wait   # expect three ~30fps lines
```

DKMS rebuilds on every kernel update (headers required). Each camera exposes
two `/dev/video` nodes — capture + metadata; check `v4l2-ctl --list-devices`,
and note replugging renumbers them.