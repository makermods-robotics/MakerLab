# Installing MakerLab (macOS & Ubuntu/Jetson)

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

## Choose your install

Two supported flavors — pick by whether you want to **run** MakerLab or **hack on**
it. They can even coexist on one machine (see the precedence note below).

- **Tool install** — `uv tool install <local-path-or-git-url>`. One command,
  no clone, a global `makerlab`/`makerlabs`/`makerlab-station` on your PATH. The
  built frontend ships inside the wheel (`frontend/dist`), so a tool install is
  a fully runnable app — nothing to build. This is the way to just *use*
  MakerLab. Trade-offs:
  - It builds a **second full environment** (torch + lerobot duplicated in
    `~/.local/share/uv/tools/`). On a Jetson wanting CUDA torch this fights the
    GPU-torch install ordering — those stations should prefer the from-source
    setup and follow [JETSON_SETUP.md](JETSON_SETUP.md).
  - It is **non-editable**: the code is a snapshot, frozen at install time. It
    does **not** track the repo — upgrade explicitly (see below).
  - `--dev` (Vite HMR) is unavailable: the frontend *source* and
    `package.json` aren't in the wheel, only the built `dist/`. Running
    `makerlab --dev` from a tool install fails fast and tells you to use a
    checkout.

  ```bash
  # from a local clone (dependable — no network, works on throttled links):
  uv tool install /path/to/MakerLab
  # or straight from git (needs github reachable; unreliable behind the GFW):
  uv tool install "git+https://github.com/makermods-robotics/MakerLab"

  makerlab                 # global command, runs from any directory
  # upgrade to newer code — re-pull the source, then:
  uv tool install --reinstall /path/to/MakerLab
  ```

- **From-source setup** — clone + editable venv (`uv pip install -e .`). This
  is what you do to **develop** MakerLab: edits to `makerlab/` take effect on the
  next run, and `makerlab --dev` gives you Vite HMR + `uvicorn --reload`. On first
  launch the entry points self-install onto your PATH (see below). This is the
  flow the platform sections document next.

## macOS

```bash
git clone https://github.com/makermods-robotics/MakerLab && cd MakerLab
git lfs install --local && git lfs pull
uv venv --python 3.12
uv pip install -e .
.venv/bin/makerlab          # prod: serves built frontend on :8000
.venv/bin/makerlab --dev    # dev: Vite HMR on :8080 + uvicorn --reload
```

**When hacking on MakerLab, invoke through `.venv/bin/...`.** A bare `makerlab` on
your PATH may resolve to a [tool install](#choose-your-install) — a frozen
snapshot that won't reflect your edits — rather than this checkout's venv;
explicit venv paths cannot lie. If a fix "isn't working", check where the
command actually points (`which makerlab`, then `readlink` it: a target under
`~/.local/share/uv/tools/` is the tool install) and what's serving :8000
(`lsof -i :8000`). To hand PATH back to the checkout, `uv tool uninstall
makerlab` and re-run `.venv/bin/makerlab` once to re-link.

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

# run headless — --lan binds 0.0.0.0, --offline sets HF_HUB_OFFLINE=1:
.venv/bin/makerlab --lan --offline          # or: .venv/bin/makerlab-station
# browse from any LAN machine: http://<jetson-ip>:8000
```

**Boot-to-robot (recommended for a permanent station):** install the
systemd unit so the server starts at power-on and restarts on crashes — no
shell, no typing:

```bash
sudo cp deploy/makerlab-station.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now makerlab-station
journalctl -u makerlab-station -f           # the server log
```

The unit runs `makerlab-station` (the offline posture) as user `makermods` —
edit the paths/user in the file if yours differ, and see the comments in
[deploy/makerlab-station.service](deploy/makerlab-station.service) for the
online-posture variant and day-to-day systemctl commands.

**Global commands without `.venv/bin/` prefixes:** three entry points live in
the venv. `makerlab` and its house name `makerlabs` are **the same command** —
run MakerLab in the friendly default posture (serve on localhost, open a browser,
Hub reachable). `makerlab-station` is the headless offline station posture
(`--lan --offline`; see below). They **self-install onto your PATH on first
launch**: any run by full path (e.g. `.venv/bin/makerlab`) symlinks all three
into `~/.local/bin` — global command, no duplicated environment, always as
fresh as the venv. From then on, bare `makerlab` works from any directory. The step is
idempotent, repoints stale links from old clones, never clobbers a
non-symlink, and can be disabled with `MAKERLAB_NO_PATH_LINK=1`. The manual
equivalent, if you prefer to do it yourself:

```bash
mkdir -p ~/.local/bin
ln -sf ~/MakerLab/.venv/bin/makerlab ~/MakerLab/.venv/bin/makerlab-station \
       ~/MakerLab/.venv/bin/makerlabs ~/.local/bin/
```

**Coexisting with a tool install.** A `uv tool install` (see [Choose your
install](#choose-your-install)) puts its *own* symlinks in the same
`~/.local/bin`, resolving into `~/.local/share/uv/tools/`. The self-link step
recognises those and **leaves them alone** — it will not overwrite a tool
install with a venv link. If both flavors are present for the same name, the
one already on your PATH wins and the launcher logs which it is; pick
deliberately with `uv tool uninstall makerlab` (to prefer this checkout) or
`MAKERLAB_NO_PATH_LINK=1` (to keep the tool install and silence the self-link).

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
that launches the server too — makerlab's in-app policy installer inherits it:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

**Problem:** huggingface.co is unreachable (downloads hang, `hf auth login`
fails).
**Solution:** split the work by what actually needs huggingface.co. The
tunnel (below) is only for writes and auth — `hf auth login`/`whoami`,
dataset/model uploads, HF Jobs, git pushes. Public bulk *downloads* should
bypass the tunnel via a read-only mirror: faster and it saves proxy quota.

```bash
export HF_ENDPOINT=https://hf-mirror.com   # downloads only; mirrors can't write
```

Three knobs, by job, and they override in this order:

- `HF_HUB_OFFLINE=1` beats everything — every Hub call fails fast instead of
  hanging (all hardware flows work offline). An explicit download needs a
  shell with this *unset* and the mirror/proxy set.
- `HF_ENDPOINT=https://hf-mirror.com` — downloads. No mirror supports writes.
- proxy vars (below) — the one path for uploads/auth/Jobs/pushes.

LAN-seed the caches (below) for weights you already have on another box.

**Problem:** you can't tell whether a Hub failure is your app or a dead proxy
relay.
**Solution:** don't mix postures on the robot station — pick one per session
and restart between them.

- *Hardware session* (teleop/record/replay/inference): `HF_HUB_OFFLINE=1`,
  **no** proxy vars. Deterministic — a dying relay node can never masquerade
  as an app bug.
- *Upload/Hub session*: proxy vars set, `HF_HUB_OFFLINE` unset.

**Problem:** the `hf` CLI feels sluggish over the tunnel.
**Solution:** expected — every API call rides the relay's round-trip, and
sequential-call tools (the `hf` CLI) feel it most. Pick a low-latency node
(the client's latency test) for interactive work; leave bulk downloads on the
mirror, off the tunnel.

**Problem:** ping succeeds but HTTPS to the same host dies, so you conclude
the network is fine.
**Solution:** ping is not a connectivity test on filtered networks. Blocking
is TLS/SNI-based — ICMP passes while HTTPS is dropped mid-handshake. Test the
actual protocol, not reachability:

```bash
curl -4 -sS --max-time 10 https://huggingface.co   # -sS surfaces the error; -s hides it
```

And beware the wrong-domain trap: `huggingface.com` is a *different*,
unblocked domain from `huggingface.co` — a working curl to `.com` proves
nothing about `.co`.

**Problem:** one machine needs to route through a restricted network via a
proxy client.
**Solution:** git, pip, uv, and `hf` all honor the standard variables. You do
**not** need the client's TUN / virtual-NIC mode (which requires a privileged
service) — its local mixed port is enough for dev tooling. Find the real port
(a Clash-family default is 7897, older builds 7890; the 9090 control API is
*not* it):

```bash
ss -tlnp | grep -iE 'clash|mihomo'   # read the mixed/http listener port
```

The GUI's "system proxy" toggle only affects GUI apps — a terminal **always**
needs explicit env vars. Exclude the LAN so robot traffic never detours into
the tunnel:

```bash
export https_proxy=http://127.0.0.1:<port> http_proxy=http://127.0.0.1:<port>
export no_proxy=localhost,127.0.0.1,192.168.0.0/16
```

Prove routing actually reaches the exit before trusting it — the returned
country code should be the exit's, not yours:

```bash
curl -4 -sS --max-time 10 https://ipinfo.io/country
```

**Problem:** the proxy replies `CONNECT ... 200` but the request then dies
with `SSL routines::unexpected eof while reading`.
**Solution:** the local proxy is fine; its upstream relay node is dead. In
the client, run the latency test over all nodes and switch to a live one. If
*every* node gives the identical error, node-switching isn't touching the
request — check two things:

- **Routing mode** — in Rule mode the domain may be sent DIRECT (never
  through any node). Retest in Global mode.
- **System clock** — `date`. TLS-based proxy protocols fail identically on
  clock skew.

**Problem:** a server you started earlier ignores the proxy/`HF_*` vars you
just exported.
**Solution:** a process only ever sees the environment it was *born* with —
exporting vars in your shell can't reach an already-running server. Verify
what it actually has, then restart it from an equipped shell:

```bash
cat /proc/<pid>/environ | tr '\0' '\n' | grep -iE 'proxy|HF_'
```

`~/.bashrc` covers new interactive logins automatically, but **not**
one-shot `ssh host 'cmd'` (Ubuntu's bashrc exits early for non-interactive
shells) and **not** systemd services (put env in the unit file).

**Problem:** installing this repo dies inside the `lerobot` dependency (a
~250 MB-history git pin) even with HTTP/1.1.
**Solution:** shallow-fetch exactly the pinned commit (SHA from
`pyproject.toml`) and install around the URL — full recipe in
[JETSON_SETUP.md](JETSON_SETUP.md#network-gotchas).

### Always-on proxy on a headless box

GUI autostart works but ties the tunnel to a logged-in desktop session — no
good on a headless robot station. The robust recipe runs the proxy *core*
(mihomo) as a systemd service: no desktop, survives reboots, auto-picks a
live node.

> Recipe, not battle-tested end-to-end in our session — verify on your box.
> Only one process may own the mixed port, so **stop the GUI client first**.

Config (`~/.config/mihomo/config.yaml`) — subscription via `proxy-providers`,
a `url-test` group that auto-selects the fastest node, and LAN/CN kept off the
tunnel:

```yaml
mixed-port: 7897
bind-address: 127.0.0.1          # local only; not exposed to the LAN
mode: rule
proxy-providers:
  sub:
    type: http
    url: "<your-subscription-url>"
    interval: 3600
    path: ./providers/sub.yaml
proxy-groups:
  - name: auto
    type: url-test
    use: [sub]
    url: http://www.gstatic.com/generate_204
    interval: 300
rules:
  - GEOIP,CN,DIRECT
  - MATCH,auto
```

Unit (`/etc/systemd/system/mihomo.service`):

```ini
[Unit]
Description=mihomo proxy core
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/mihomo -d /home/<user>/.config/mihomo
Restart=always
User=<user>

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now mihomo
curl -4 -sS --max-time 10 https://ipinfo.io/country   # confirm the exit
```

Remember services don't read `~/.bashrc` — consumers still need the
`https_proxy`/`http_proxy` env vars pointing at `127.0.0.1:7897`.

### Skip the internet: LAN seeding

A second machine that already has the bits beats any mirror. These are plain
directory copies — lerobot/makerlab discover content by scanning; no
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
