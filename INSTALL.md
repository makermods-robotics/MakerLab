# Installing LeLab (macOS & Ubuntu/Jetson)

Battle-tested install paths for the two machines we actually run: a macOS
dev/operator machine and a headless Ubuntu box (NVIDIA Jetson) wired to the
arms. The second half is the accumulated bag of performance and
restricted-network tricks — each one earned. Deep Jetson-specific gotchas
(GPU torch vs Python versions, serial permissions, LAN serving) live in
[JETSON_SETUP.md](JETSON_SETUP.md); this guide covers the happy path plus the
speed-ups.

## Prerequisites (both platforms)

- **git-lfs, initialized in the clone** — required before *building* the
  frontend (not for serving): `frontend/public/**` meshes are LFS-tracked,
  and building from an un-smudged clone silently bakes 131-byte pointer
  files into `dist/`, breaking the 3D viewer:

  ```bash
  git lfs install --local && git lfs pull
  # sanity: this must print binary junk, NOT "version https://git-"
  head -c 20 frontend/public/so-101-urdf/meshes/base_so101_v2.stl
  ```

- **uv** — provides Python 3.12 on systems that don't ship it and makes the
  big dependency resolve fast:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc   # or restart the shell: ~/.local/bin joins PATH at login
  ```

## macOS

```bash
git clone https://github.com/makermods-robotics/MakerLab && cd MakerLab
git lfs install --local && git lfs pull
uv venv --python 3.12
uv pip install -e .
.venv/bin/lelab          # prod: serves built frontend on :8000
.venv/bin/lelab --dev    # dev: Vite HMR on :8080 + uvicorn --reload
```

**Always invoke through `.venv/bin/...`.** A `lelab` on the bare PATH may be
a stale `uv tool` snapshot from months ago; explicit venv paths cannot lie.
When a fix "isn't working", first check what's actually serving :8000
(`lsof -i :8000`).

macOS ships `openrsync`, not GNU rsync — use `--progress`, not
`--info=progress2`, and `ssh <host> "mkdir -p <dir>"` before rsyncing into a
path that doesn't exist yet (no `--mkpath`).

## Ubuntu / Jetson (headless robot station)

```bash
# one-time system prep
sudo usermod -aG dialout $USER              # serial ports; RE-LOGIN required
sudo systemctl disable --now ModemManager   # stops AT-command probes on the servo bus
sudo apt install -y git git-lfs ffmpeg

git clone https://github.com/makermods-robotics/MakerLab && cd MakerLab
uv venv --python 3.12                       # JetPack 6 ships 3.10; uv fetches 3.12
uv pip install -e .                         # GPU inference? read JETSON_SETUP.md FIRST

# run headless — the `lelab` launcher binds 127.0.0.1 only:
HF_HUB_OFFLINE=1 .venv/bin/uvicorn lelab.server:app --host 0.0.0.0 --port 8000
# browse from any LAN machine: http://<jetson-ip>:8000
```

The `usermod` only takes effect on a fresh login, and the server inherits its
groups from the shell that launched it — log out, back in, *then* start the
server. `HF_HUB_OFFLINE=1` makes every Hub-touching code path fail fast
instead of hanging when huggingface.co is unreachable; teleop, recording,
calibration, and inference from local checkpoints are fully functional
without it ever being reachable.

## The performance & restricted-network bag of tricks

Everything below was used in anger. Symptoms first, so you can grep your way
here.

### Network layer 1: the LAN itself

**WiFi power save** — symptom: LAN pings swinging 12–232 ms, scp at dial-up
speed, jittery UI, despite strong signal. The radio naps between beacons and
packets queue at the AP. Robot servers should never nap:

```bash
sudo iw dev <iface> set power_save off                      # instant, until reboot
echo -e "[connection]\nwifi.powersave = 2" | \
  sudo tee /etc/NetworkManager/conf.d/wifi-powersave.conf   # permanent (2 = disable)
```

Check link quality while you're there: `iw dev <iface> link` — signal
should be better than −60 dBm, and a TX bitrate far below RX suggests a
congested band (prefer the 5 GHz SSID over 2.4 GHz).

### Network layer 2: reaching the internet

**IPv6 blackhole** — symptom: `Failed to connect ... port 443 after ~130s`
while the network "works". The router advertises IPv6 it can't route; AAAA
gets tried first and eats ~65 s × 2. Diagnose in 10 seconds
(`curl -4` works, `curl -6` hangs), fix system-wide:

```bash
echo "precedence ::ffff:0:0/96 100" | sudo tee -a /etc/gai.conf
```

Harmless on healthy networks — safe to bake into provisioning.

**Mid-transfer resets on large fetches** — symptom:
`RPC failed; curl 16 Error in the HTTP2 framing layer` partway through a big
clone. HTTP/1.1 is far more tolerant of hostile middleboxes:

```bash
git config --global http.version HTTP/1.1
```

**Throttled-path triage** — before fighting a slow transfer, measure it:

```bash
curl -4 -o /dev/null -sw 'connect: %{time_connect}s TLS: %{time_appconnect}s speed: %{speed_download} B/s\n' <url>
```

Healthy connect/TLS with speed in single-digit KB/s means the *path* is
throttled — no client-side flag fixes that; use a mirror, a proxy, or the
LAN tricks below.

**PyPI mirror** — turns minutes-per-package into seconds on slow
international links (TUNA is a full mirror):

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

Set them in the shell that launches the server too — lelab's in-app policy
installer inherits them (it auto-detects uv-venvs and installs with
`uv pip install --python <venv>`).

**Hugging Face when huggingface.co is unreachable** —
`HF_ENDPOINT=https://hf-mirror.com` covers model/dataset *downloads*;
uploads need a real tunnel. For robot work, prefer `HF_HUB_OFFLINE=1` plus
locally-seeded caches (below).

**Routing one machine through a VPN/proxy** — git, pip, uv, and `hf` all
honor the standard variables; exclude the LAN so robot traffic never
detours:

```bash
export https_proxy=http://<proxy-host>:<port> http_proxy=http://<proxy-host>:<port>
export no_proxy=localhost,127.0.0.1,192.168.0.0/16
```

**The big pinned dependency** — `lerobot` is a ~250 MB-history git pin. If
even HTTP/1.1 can't survive it, shallow-fetch exactly the pinned commit (SHA
from `pyproject.toml`) and install around the URL — see
[JETSON_SETUP.md](JETSON_SETUP.md#network-gotchas) for the full recipe.

### Skip the internet entirely: LAN seeding

A second machine that already has the bits beats any mirror. All of these
are plain directory copies — lerobot/lelab discover content by scanning, no
registration steps except where noted.

**Code (git bundle)** — deploy commits without GitHub in the loop:

```bash
# machine with the commits:
git bundle create /tmp/update.bundle <last-common-sha>..main
scp /tmp/update.bundle user@station:/tmp/
# station:
git pull /tmp/update.bundle main        # same SHAs as origin; reconciles cleanly later
```

**Datasets** — straight into the LeRobot cache, appears in the UI
immediately (no trailing slash on the source path):

```bash
rsync -a --progress ~/.cache/huggingface/lerobot/<ns>/<dataset> \
  user@station:~/.cache/huggingface/lerobot/<ns>/
```

**Policy checkpoints** — copy anywhere stable, then register (a flat dir
with `config.json` counts as one checkpoint):

```bash
rsync -a --progress <checkpoint-dir>/ user@station:~/models/<name>/
curl -X POST http://station:8000/jobs/import -H 'Content-Type: application/json' \
  -d '{"source": "/home/<user>/models/<name>", "name": "<display name>"}'
```

Don't move the directory afterwards — the import records the absolute path.

**Hub model caches** (e.g. SmolVLA's VLM backbone) — copy the `models--*`
dirs; `HF_HUB_OFFLINE=1` loads happily from a seeded cache:

```bash
rsync -a --progress ~/.cache/huggingface/hub/models--<org>--<model> \
  user@station:~/.cache/huggingface/hub/
```

**Calibrations** — plain JSON; copy
`~/.cache/huggingface/lerobot/calibration/` between machines that drive the
same physical arms.

### Serving & access tricks

**SSH tunnel = free secure context** — browser cameras (getUserMedia) are
blocked on plain-HTTP non-localhost origins, but localhost is exempt:

```bash
ssh -L 8000:127.0.0.1:8000 user@station   # then browse http://localhost:8000
```

Cameras plugged into the *server* don't need this — they stream via the
backend MJPEG previews (`/camera-preview/{index}`) and OpenCV capture.

**mDNS may lie to you** — `station.local` resolution can break (VPN DNS
capture on macOS, notably). `arp -a | grep <mac-prefix>` or the router's
client list finds the IP; prefer IPs in scripts.

**Emergency stops from anywhere on the LAN** — if a browser tab dies
mid-session, the robot keeps going; recovery is one request:

```bash
curl -X POST http://station:8000/stop-recording   # or /stop-teleoperation
```
