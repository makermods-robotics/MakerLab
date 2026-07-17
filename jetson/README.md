# Running MakerLab on an NVIDIA Jetson

This guide covers MakerLab as a self-contained Jetson robot station: SO-101
arms and cameras connected to the Jetson, with the UI opened either locally or
from another machine on the LAN. It includes the Jetson-specific installation
order, CUDA checks, headless operation, restricted-network practices, and the
multi-camera kernel patch.

For macOS, generic Linux, tool installation, and development commands, see the
main [README](../README.md).

## Supported station postures

Choose the posture before starting the server:

| Posture | Command | Hub access |
| --- | --- | --- |
| Local display | `.venv/bin/makerlab` | Enabled unless the environment says otherwise |
| LAN/headless | `.venv/bin/makerlab --lan` | Enabled |
| Deterministic offline station | `.venv/bin/makerlab-station` | Disabled with `HF_HUB_OFFLINE=1` |

The offline posture is recommended while operating hardware when internet
access is unreliable. Calibration, teleoperation, recording, replay, and
inference from local checkpoints continue to work. Login, Hub transfers, and
Hugging Face Jobs fail immediately instead of hanging.

## Quick start

The GPU installation order matters. Install the compatible CUDA PyTorch pair
before installing MakerLab, or the dependency resolver may replace it with CPU
wheels.

```bash
# System preparation
sudo usermod -aG dialout "$USER"       # log out and back in afterward
sudo systemctl disable --now ModemManager
sudo apt update
sudo apt install -y git git-lfs ffmpeg v4l-utils

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone MakerLab and materialize its LFS assets
git clone https://github.com/makermods-robotics/MakerLab
cd MakerLab
git lfs install --local
git lfs pull

# Create the Python 3.12 environment
uv venv --python 3.12

# JetPack 7.2 GPU station: install this pair before MakerLab
uv pip install "torch==2.10.0" "torchvision==0.25.0" \
  --index-url https://download.pytorch.org/whl/cu130

# Install MakerLab into the same environment
uv pip install -e .

# Start locally, or serve an offline station to the LAN
.venv/bin/makerlab
# .venv/bin/makerlab-station
```

For a CPU-only JetPack 6 station, omit the explicit CUDA PyTorch command and
run `uv pip install -e .`. Teleoperation, calibration, recording, and CPU
inference remain available.

## System preparation

### Serial-port access

The user running MakerLab must belong to `dialout`:

```bash
sudo usermod -aG dialout "$USER"
```

Group membership takes effect only after a fresh login. A server started before
that login retains the old groups and cannot open the arm ports.

Disable ModemManager so it does not send modem probes to the servo bus:

```bash
sudo systemctl disable --now ModemManager
```

### Git LFS

The SO-101 meshes in `frontend/public/` are stored with Git LFS. Before
rebuilding the frontend, verify that the clone contains real mesh data rather
than pointer files:

```bash
git lfs install --local
git lfs pull
head -c 20 frontend/public/so-101-urdf/meshes/base_so101_v2.stl
```

The final command should emit binary data, not text beginning with
`version https://git-lfs`.

## JetPack and CUDA PyTorch

GPU training and on-device GPU inference require a compatible JetPack, Python,
PyTorch, and CUDA combination.

### JetPack generations

Check the installed BSP:

```bash
head -1 /etc/nv_tegra_release
```

- JetPack 6 reports an R36 BSP and uses Ubuntu 22.04 with a Python 3.10-oriented
  GPU-wheel ecosystem. MakerLab requires Python 3.12, so use it as a CPU-only
  hardware station unless you maintain a custom GPU environment.
- JetPack 7.2 uses Ubuntu 24.04, Python 3.12, and CUDA 13 and is the supported
  GPU path described below.

### Install the pinned CUDA pair first

Install exactly the compatible PyTorch and torchvision pair before
`uv pip install -e .`:

```bash
uv pip install "torch==2.10.0" "torchvision==0.25.0" \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
```

The cu130 index also contains versions outside LeRobot's supported
`torch>=2.7,<2.11` range. Installing an unpinned version, or installing only
torch without the matching torchvision, can cause the MakerLab installation to
replace the CUDA build with CPU wheels.

Verify the final environment with a CUDA operation that exercises cuBLAS:

```bash
.venv/bin/python - <<'EOF'
import torch
import torchvision
import torch.nn as nn

print(torch.__version__, torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print(nn.Linear(64, 64).cuda()(torch.randn(8, 64, device="cuda")).sum())
EOF
```

Both package versions should include `+cu130`, CUDA should be available, and
the linear layer should run without an exception.

### Repair a mixed cuBLAS installation

If startup fails with `CUBLAS_STATUS_NOT_INITIALIZED` from
`cublasLtMatmulAlgoGetHeuristic`, the loader is probably mixing pip-bundled and
system cuBLAS libraries. Point both libraries at the system pair:

```bash
cd .venv/lib/python3.12/site-packages/nvidia/cu13/lib
for f in libcublas.so.13 libcublasLt.so.13; do
  [ -f "$f" ] && mv "$f" "$f.pip-bak"
  ln -sf "/usr/local/cuda/lib64/$f" "$f"
done
```

Return to the repository and rerun the verification above. Reinstalling
PyTorch restores the pip copies, so repeat the symlink operation afterward. A
`CUDNN_STATUS_*` failure caused by the same mixed-library condition can be
treated similarly under `nvidia/cudnn/lib`.

### Flash JetPack 7.2

JetPack 7.2 provides a bootable Jetson ISO. Write it to a USB drive of at least
16 GB, boot the Jetson from that drive, and install to NVMe. Reference Orin AGX,
NX, and Nano Super development kits require sufficiently recent QSPI firmware;
update it from the existing JetPack installation before wiping the disk if the
BSP predates r35.5.

Flashing erases the target disk. Back up at least:

- `~/.cache/huggingface/lerobot/`, including calibrations and datasets.
- Imported policy checkpoints.
- Shell configuration containing network or CUDA settings.

## Running the station

### Local display

```bash
.venv/bin/makerlab
```

MakerLab binds to `127.0.0.1:8000` and opens the browser. If Chromium cannot
create a WebGL context, the 3D viewer shows a fallback rather than crashing the
teleoperation page.

### LAN or headless operation

```bash
.venv/bin/makerlab --lan
```

Open `http://<jetson-ip>:8000` from another machine on the LAN. Use
`makerlab-station` when the station should also be offline:

```bash
.venv/bin/makerlab-station
```

Camera previews use browser media APIs and are constrained by browser secure-
context and device-locality rules. A browser on another computer sees that
computer's media devices, not the cameras physically connected to the Jetson.
For setup that depends on visually identifying Jetson cameras, use a browser on
the Jetson or provide an appropriate HTTPS deployment.

If a browser disappears while hardware is active, the stop endpoints remain
available from the LAN:

```bash
curl -X POST http://<jetson-ip>:8000/stop-recording
curl -X POST http://<jetson-ip>:8000/stop-teleoperation
curl -X POST http://<jetson-ip>:8000/stop-inference
```

### Global launcher commands

Running `.venv/bin/makerlab` attempts to symlink both MakerLab entry points into
`~/.local/bin`. It is idempotent, repoints stale symlinks from old clones, does
not overwrite regular files, and leaves commands owned by a `uv tool install`
alone.

Set `MAKERLAB_NO_PATH_LINK=1` to opt out. The manual equivalent is:

```bash
mkdir -p ~/.local/bin
ln -sf "$HOME/MakerLab/.venv/bin/makerlab" \
       "$HOME/MakerLab/.venv/bin/makerlab-station" \
       ~/.local/bin/
```

## Multi-camera USB patch

### Symptom

The third concurrent USB camera fails with one of these errors while any two
cameras work:

```text
VIDIOC_STREAMON returned -1 (No space left on device)
NotReadableError: could not start video source
```

`No space left on device` means USB isochronous bandwidth, not disk space.

### Cause

The UVC driver reserves the bandwidth advertised by the camera firmware. Some
cameras advertise an inflated worst case even when sending compressed MJPEG,
so two reservations can fill a 480 Mbps USB 2 bus. The kernel's
`FIX_BANDWIDTH` quirk computes realistic reservations but normally applies only
to uncompressed formats.

MakerLab vendors a Linux 6.8.12 `uvcvideo` source tree with the compressed-
format gate removed from `uvc_fixup_video_ctrl()`:

```diff
- if (!(format->flags & UVC_FMT_FLAG_COMPRESSED) &&
-     stream->dev->quirks & UVC_QUIRK_FIX_BANDWIDTH &&
+ if (stream->dev->quirks & UVC_QUIRK_FIX_BANDWIDTH &&
      stream->intf->num_altsetting > 1) {
```

### Install

```bash
sudo jetson/uvcvideo-mjpg/install.sh
```

The installer:

1. Installs DKMS and the running kernel's headers.
2. Copies the vendored source into `/usr/src`.
3. Builds and installs the module through DKMS.
4. Enables `quirks=128` in `/etc/modprobe.d/uvcvideo.conf`.
5. Reloads the driver when no camera is active.

The installer warns before building against a non-6.8 kernel. A driver built
from mismatched source can fail or misbehave; refresh the vendored source from
the target kernel before accepting that risk.

See [uvcvideo-mjpg/README.md](uvcvideo-mjpg/README.md) for the component-level
patch notes.

### Verify

Close other camera applications, identify the capture nodes with
`v4l2-ctl --list-devices`, and run three streams concurrently:

```bash
for d in 0 2 4; do
  v4l2-ctl -d "/dev/video$d" \
    --set-fmt-video=width=640,height=480,pixelformat=MJPG \
    --stream-mmap --stream-count=100 --stream-to=/dev/null &
done
wait
```

Each stream should report approximately 30 FPS. Cameras often expose separate
capture and metadata nodes, and reconnecting a camera can renumber them.

To remove the patch:

```bash
sudo dkms remove uvcvideo-mjpg/6.8.12 --all
sudo rm /etc/modprobe.d/uvcvideo.conf
sudo depmod -a
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo
```

## Network troubleshooting

### Wi-Fi latency

Robot stations should not use Wi-Fi power saving:

```bash
sudo iw dev <interface> set power_save off
echo -e "[connection]\nwifi.powersave = 2" | \
  sudo tee /etc/NetworkManager/conf.d/wifi-powersave.conf
```

Check the link with `iw dev <interface> link`. Prefer 5 GHz or wired Ethernet
when latency varies sharply despite a strong signal.

### Broken IPv6 routing

If HTTPS stalls but `curl -4` succeeds, the network may advertise IPv6 without
routing it correctly. Prefer IPv4-mapped addresses:

```bash
echo "precedence ::ffff:0:0/96 100" | sudo tee -a /etc/gai.conf
```

### Unreliable Git transport

Some middleboxes break large HTTP/2 Git transfers. Force HTTP/1.1:

```bash
git config --global http.version HTTP/1.1
```

If GitHub remains unavailable, transfer repository history with a Git bundle
from another machine:

```bash
# Connected machine
git bundle create /tmp/makerlab-update.bundle main
scp /tmp/makerlab-update.bundle user@<jetson-ip>:/tmp/

# Jetson checkout
git pull /tmp/makerlab-update.bundle main
```

### Python package mirrors

`uv` and pip honor the standard package-index environment variables:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

Start MakerLab from a shell containing these variables when an in-app extra
installer should use the same mirror.

### Hugging Face access

Use one deliberate posture per session:

- Hardware session: `HF_HUB_OFFLINE=1`, with no proxy dependency.
- Hub session: unset `HF_HUB_OFFLINE`, configure the mirror or proxy, and
  restart MakerLab from that environment.

For public downloads through a read-only mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Mirrors cannot perform authentication, uploads, repository writes, or Hugging
Face Jobs. Those operations require direct access or a proxy:

```bash
export https_proxy=http://127.0.0.1:<port>
export http_proxy=http://127.0.0.1:<port>
export no_proxy=localhost,127.0.0.1,192.168.0.0/16
```

Environment changes do not affect an already-running server. Stop it and
restart it from the configured shell. Test HTTPS itself rather than relying on
ping:

```bash
curl -4 -sS --max-time 10 https://huggingface.co
```

### Seed data over the LAN

Copying an existing cache is often faster and more reliable than downloading
again.

Datasets:

```bash
rsync -a --progress \
  ~/.cache/huggingface/lerobot/<namespace>/<dataset> \
  user@<jetson-ip>:~/.cache/huggingface/lerobot/<namespace>/
```

Hub model caches:

```bash
rsync -a --progress \
  ~/.cache/huggingface/hub/models--<organization>--<model> \
  user@<jetson-ip>:~/.cache/huggingface/hub/
```

Calibration files are plain JSON. Copy
`~/.cache/huggingface/lerobot/calibration/` only between machines that operate
the same physical arms.

Imported checkpoints can be copied to a stable directory and added through the
MakerLab model library's **Import from disk** action. Do not move the directory
after importing a checkpoint by reference.

## Operational checks

Before a hardware session:

```bash
groups                         # must include dialout
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
v4l2-ctl --list-devices
curl -fsS http://127.0.0.1:8000/health
```

For server logs, run MakerLab in a terminal or capture its stdout and stderr
with the process supervisor used on the station.
