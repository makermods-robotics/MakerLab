# Running LeLab on an NVIDIA Jetson

Notes from bringing LeLab up on a headless Jetson used as a robot station
(arms + cameras plugged into the Jetson, UI opened from a laptop browser on
the same LAN). Everything below was hit in practice; the happy path hides
none of it.

## TL;DR checklist

```bash
# --- one-time system prep ---
sudo usermod -aG dialout $USER          # serial port access (re-login required!)
sudo systemctl disable --now ModemManager   # stop it probing the arms as modems
sudo apt install -y git ffmpeg
echo "precedence ::ffff:0:0/96 100" | sudo tee -a /etc/gai.conf   # if IPv6 is flaky, see below
git config --global http.version HTTP/1.1                          # if large fetches die, see below

# --- install ---
curl -LsSf https://astral.sh/uv/install.sh | sh     # then: source ~/.bashrc
git clone <this-repo> && cd <repo>
uv venv --python 3.12                                # JetPack 6 ships 3.10; uv downloads 3.12
uv pip install -e .                                  # GPU inference: install Jetson torch FIRST, see below

# --- run (headless) ---
.venv/bin/lelab --lan --offline
# browse from the laptop: http://<jetson-ip>:8000
```

## Python version vs. GPU torch — decide before installing

LeLab and its pinned `lerobot` both require **Python ≥ 3.12**, but JetPack 6
(Ubuntu 22.04) ships 3.10 and the community GPU torch wheels for JetPack 6
(`pypi.jetson-ai-lab.io/jp6/...`) are **cp310 only**. These constraints are
incompatible — pick a lane:

- **Teleop / recording / calibration station (no on-device inference):**
  Python 3.12 venv via uv + CPU torch from PyPI (the default resolution).
  Everything hardware-related works; the GPU just idles. This is the
  fast path and is *not* blocked by anything below.
- **On-device policy inference on JetPack 6:** there is no clean fit.
  Options are an NVIDIA container with its own torch, or staying on
  Python 3.10 and overriding both packages' `requires-python` (untested,
  not recommended).
- **On-device policy inference, properly:** flash **JetPack 7.2+**
  (Ubuntu 24.04 — supports Orin AGX/NX/Nano Super as of mid-2026). Python
  3.12 is the system Python and mainstream CUDA aarch64 wheels work:
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`.
  Check which generation you're on with `head -1 /etc/nv_tegra_release`
  (R36 = JetPack 6, R38/R39 = JetPack 7).

  **Flashing 7.2 needs no host PC**: JetPack 7.2 ships a bootable **Jetson
  ISO** — write it to a ≥16 GB USB stick (balenaEtcher; SD-card images are
  discontinued in 7.2), boot the Jetson from the stick, install onto the
  NVMe. Works on the reference devkits (Orin AGX/NX/Nano Super) as long as
  the installed firmware is JetPack-6 generation (BSP ≥ r35.5) — check with
  `head -1 /etc/nv_tegra_release` before starting, and update QSPI firmware
  from your current JetPack first if it's older. It's a full disk wipe:
  evacuate `~/.cache/huggingface/lerobot/` (calibrations + datasets),
  imported checkpoints, and `~/.bashrc` beforehand.

**Install order matters for GPU torch:** install the Jetson/CUDA torch wheel
into the venv *before* `uv pip install -e .`, or dependency resolution pulls
generic CPU torch over it. Verify with
`python -c "import torch; print(torch.cuda.is_available())"`.

## Network gotchas

The `lerobot` dependency is a **git pin** (see `pyproject.toml`), and its
repo is ~250 MB of history — much bigger than this repo. On a slow, filtered,
or unreliable network the install fails in several distinct ways:

- **`Failed to connect to github.com port 443 after ~130s`** — usually the
  network advertises IPv6 but blackholes it, and the AAAA record gets tried
  first (~65 s × 2). Confirm with `curl -4` (works) vs `curl -6` (hangs),
  then prefer IPv4 system-wide:
  `echo "precedence ::ffff:0:0/96 100" | sudo tee -a /etc/gai.conf`.
  Takes effect immediately, harmless on healthy networks.
- **`RPC failed; curl 16 Error in the HTTP2 framing layer` mid-fetch** —
  connection reset during a large transfer.
  `git config --global http.version HTTP/1.1` is markedly more tolerant.
- **Still dying?** Skip the big fetch entirely: shallow-fetch the pinned
  commit (grab the SHA from `pyproject.toml`) and install around the git URL:

  ```bash
  git init lerobot && cd lerobot
  git remote add origin https://github.com/huggingface/lerobot.git
  git fetch --depth 1 origin <pinned-sha>
  git checkout FETCH_HEAD && cd ..
  uv pip install -e "./lerobot[core_scripts,feetech,training]"
  uv pip install --no-deps -e ./<this-repo>
  uv pip install "fastapi[standard]" websockets uvicorn psutil
  ```

- **huggingface.co unreachable** (some networks block it): hardware flows
  don't need it. Launch with `HF_HUB_OFFLINE=1` so nothing ever sits in a
  connect timeout; `HF_ENDPOINT=https://hf-mirror.com` covers model/dataset
  *downloads*; uploads need a proxy (`https_proxy=...` is honored by git,
  pip, and `hf`) or do them from another machine.

## Serial ports

- Arms appear as `/dev/ttyACM0..N` (not macOS-style `/dev/tty.usbmodem*`).
  The in-app unplug/replug port detection handles this correctly.
- **`Could not connect on port '/dev/ttyACMx'` almost always means
  permissions**, not a wrong port: you must be in the `dialout` group
  (`crw-rw---- root dialout`). After `sudo usermod -aG dialout $USER`,
  **log out and back in** — group membership is stamped at login — and
  restart the LeLab server from the *new* session, or the server process
  keeps the old groups.
- **Disable ModemManager.** Stock Ubuntu probes every new ACM device with
  AT commands for ~15–30 s after plug-in, holding the port busy — on a bus
  with live servos. `sudo systemctl disable --now ModemManager`.
- **ACM numbering is a boot-order lottery.** With multiple arms (bimanual =
  4 devices), pin each arm to a stable name by its USB serial. Get serials
  with `udevadm info -a -n /dev/ttyACM0 | grep -i '{serial}' | head -2`,
  then e.g. `/etc/udev/rules.d/99-so101.rules`:

  ```
  SUBSYSTEM=="tty", ATTRS{serial}=="<serial-of-leader>",  SYMLINK+="so101_leader"
  SUBSYSTEM=="tty", ATTRS{serial}=="<serial-of-follower>", SYMLINK+="so101_follower"
  ```

  Reload with `sudo udevadm control --reload && sudo udevadm trigger`, then
  select the `/dev/so101_*` names in the UI.

## Serving the UI over the LAN

- Run the launcher with `--lan` (`.venv/bin/lelab --lan`) to bind 0.0.0.0
  and skip the open-a-local-browser step; add `--offline` for
  `HF_HUB_OFFLINE=1`. (Without `--lan`, the launcher binds 127.0.0.1 only —
  useless headless.) The committed `frontend/dist/` is served at `/`, so no
  Node/npm is needed on the Jetson.
- **You need a bundle that includes the LAN-hosting fixes** (commit
  `b762d2c` or later). Older bundles render a blank page from any
  non-localhost origin (`crypto.randomUUID` is a secure-context-only API)
  and send API calls to a hardcoded `localhost:8000` (i.e. to the *viewing*
  machine). If you see a blank page, check the served JS for those two bugs
  before debugging anything else.
- **Browser-side cameras (laptop webcam, phone) need a secure context**, so
  plain `http://<jetson-ip>:8000` can't use them (`getUserMedia` is blocked).
  Either set up HTTPS (see [HTTPS_SETUP.md](HTTPS_SETUP.md) /
  [PHONE_CAMERA_SETUP.md](PHONE_CAMERA_SETUP.md)) or tunnel:
  `ssh -L 8000:127.0.0.1:8000 <user>@<jetson>` and browse
  `http://localhost:8000` — localhost counts as secure, cameras work with
  zero cert setup. USB cameras plugged into the Jetson go through OpenCV
  on the backend and don't care about any of this.
- No display manager conflicts: the `webbrowser.open()` call in the
  launcher is harmless when headless, but you're not using the launcher
  anyway.

## Assorted

- **git-lfs is NOT needed to *serve*** — the STL meshes committed under
  `frontend/dist/` are real binaries by design (`dist/** -filter` in
  `.gitattributes`). It **is** needed before *building* the frontend in any
  clone: `frontend/public/**` meshes are LFS-tracked, and building from an
  un-smudged clone silently copies 131-byte pointer files into `dist/`,
  breaking the 3D viewer. `git lfs install --local && git lfs pull` first.
- Calibration JSONs (`~/.cache/huggingface/lerobot/calibration/`) are plain
  files — `scp` them from another machine to skip recalibrating the same
  physical arms.
- Saved ports (`~/.cache/huggingface/lerobot/ports/`) are per-machine and
  will be macOS paths if copied — let the UI re-detect instead.
- If the browser tab is closed mid-recording, the session keeps running
  headless with torque on (no page-leave stop yet). Recovery:
  `curl -X POST http://<jetson-ip>:8000/stop-recording`.
- Consider a systemd unit for the uvicorn command so the station survives
  reboots.
