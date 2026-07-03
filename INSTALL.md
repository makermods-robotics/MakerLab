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

## Troubleshooting: problem → solution

Everything below was hit in practice. Find your symptom, apply the fix.

### LAN & wifi

**Problem:** LAN pings swing 12–232 ms, scp crawls, the UI feels jittery —
despite strong wifi signal.
**Solution:** disable wifi power save (the radio naps between beacons and
packets queue at the AP; robot servers should never nap):

```bash
sudo iw dev <iface> set power_save off                      # instant, until reboot
echo -e "[connection]\nwifi.powersave = 2" | \
  sudo tee /etc/NetworkManager/conf.d/wifi-powersave.conf   # permanent (2 = disable)
```

**Problem:** wifi is still slow/jittery after that.
**Solution:** check the link — `iw dev <iface> link`. Signal should be
better than −60 dBm; a TX bitrate far below RX means a congested band —
move to the 5 GHz SSID instead of 2.4 GHz, or run a cable.

**Problem:** `station.local` doesn't resolve (often when a VPN captures DNS
on macOS).
**Solution:** find the IP with `arp -a | grep <mac-prefix>` or the router's
client list; prefer raw IPs in scripts.

### Reaching the internet

**Problem:** `Failed to connect ... port 443 after ~130s` while the network
otherwise "works".
**Solution:** the router advertises IPv6 it can't route; the AAAA record is
tried first and eats ~65 s × 2. Confirm with `curl -4 <url>` (works) vs
`curl -6 <url>` (hangs), then prefer IPv4 system-wide — harmless on healthy
networks, safe to bake into provisioning:

```bash
echo "precedence ::ffff:0:0/96 100" | sudo tee -a /etc/gai.conf
```

**Problem:** `RPC failed; curl 16 Error in the HTTP2 framing layer` partway
through a big clone/fetch.
**Solution:** drop git to HTTP/1.1 — far more tolerant of hostile
middleboxes:

```bash
git config --global http.version HTTP/1.1
```

**Problem:** a transfer is slow and you don't know which layer to blame.
**Solution:** measure before fighting:

```bash
curl -4 -o /dev/null -sw 'connect: %{time_connect}s TLS: %{time_appconnect}s speed: %{speed_download} B/s\n' <url>
```

Healthy connect/TLS + single-digit KB/s speed = the *path* is throttled; no
client flag fixes that. Use a mirror, a proxy, or LAN seeding (below).

**Problem:** pip/uv installs take minutes per package.
**Solution:** use a domestic full PyPI mirror (TUNA). Set it in the shell
that launches the server too — lelab's in-app policy installer inherits it:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

**Problem:** huggingface.co is unreachable (downloads hang, `hf auth login`
fails).
**Solution:** three tools, by job — `HF_ENDPOINT=https://hf-mirror.com` for
model/dataset *downloads*; `HF_HUB_OFFLINE=1` on the robot server so every
Hub call fails fast instead of hanging (all hardware flows work offline);
LAN-seed the caches (below) for weights you already have elsewhere. Uploads
need a real tunnel.

**Problem:** one machine needs to route through a VPN/proxy.
**Solution:** git, pip, uv, and `hf` all honor the standard variables;
exclude the LAN so robot traffic never detours:

```bash
export https_proxy=http://<proxy-host>:<port> http_proxy=http://<proxy-host>:<port>
export no_proxy=localhost,127.0.0.1,192.168.0.0/16
```

**Problem:** installing this repo dies inside the `lerobot` dependency (a
~250 MB-history git pin) even with HTTP/1.1.
**Solution:** shallow-fetch exactly the pinned commit (SHA from
`pyproject.toml`) and install around the URL — full recipe in
[JETSON_SETUP.md](JETSON_SETUP.md#network-gotchas).

### Skip the internet: LAN seeding

A second machine that already has the bits beats any mirror. These are plain
directory copies — lerobot/lelab discover content by scanning; no
registration step except where noted.

**Problem:** the station can't pull code from GitHub (blocked/throttled).
**Solution:** ship commits as a git bundle — same SHAs as origin, so a later
real `git pull` reconciles cleanly:

```bash
# machine with the commits:
git bundle create /tmp/update.bundle <last-common-sha>..main
scp /tmp/update.bundle user@station:/tmp/
# station:
git pull /tmp/update.bundle main
```

**Problem:** a dataset exists on machine A, needed on machine B.
**Solution:** rsync into the LeRobot cache (no trailing slash on the
source); it appears in the UI immediately:

```bash
rsync -a --progress ~/.cache/huggingface/lerobot/<ns>/<dataset> \
  user@station:~/.cache/huggingface/lerobot/<ns>/
```

**Problem:** a trained policy checkpoint needs to run on the station.
**Solution:** copy it somewhere stable, then register it (a flat dir with
`config.json` counts as one checkpoint). Don't move the directory afterwards
— the import records the absolute path:

```bash
rsync -a --progress <checkpoint-dir>/ user@station:~/models/<name>/
curl -X POST http://station:8000/jobs/import -H 'Content-Type: application/json' \
  -d '{"source": "/home/<user>/models/<name>", "name": "<display name>"}'
```

**Problem:** a policy needs Hub weights (e.g. SmolVLA's VLM backbone) and
the station can't reach the Hub.
**Solution:** seed the hub cache from a machine that has it;
`HF_HUB_OFFLINE=1` loads happily from a seeded cache:

```bash
rsync -a --progress ~/.cache/huggingface/hub/models--<org>--<model> \
  user@station:~/.cache/huggingface/hub/
```

**Problem:** recalibrating arms that were already calibrated on another
machine.
**Solution:** don't — calibrations are plain JSON; copy
`~/.cache/huggingface/lerobot/calibration/` between machines that drive the
same physical arms.

### Serving & access

**Problem:** browser cameras (getUserMedia) are blocked on
`http://<station-ip>:8000`.
**Solution:** localhost is exempt from the secure-context rule — tunnel and
browse `http://localhost:8000`:

```bash
ssh -L 8000:127.0.0.1:8000 user@station
```

Cameras plugged into the *server* don't need this at all — they stream via
the backend MJPEG previews (`/camera-preview/{index}`) and OpenCV capture.

**Problem:** the browser tab died mid-session and the robot is still going,
torque on.
**Solution:** the stop endpoints work from any machine on the LAN:

```bash
curl -X POST http://station:8000/stop-recording   # or /stop-teleoperation
```
