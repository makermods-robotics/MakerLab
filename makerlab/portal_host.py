# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SERVER-side LiveKit Portal host — see docs/remote-portal/SPEC.md.

This module runs on the machine that owns the follower arm + cameras (the
OpenBooth station). It owns three things:

1. the `livekit-server` subprocess (config generated at runtime under the
   lerobot cache dir, stdout/stderr tee'd to a log file there),
2. the LiveKit API key/secret — they never leave this machine; the client only
   ever receives a minted JWT from `POST /portal/token`,
3. the `/portal/*` HTTP handlers the router in server.py calls.

Everything degrades gracefully: a machine without the `livekit-server` binary
or without the `livekit` Python packages still runs MakerLab normally, and
`/portal/info` reports `{"enabled": false, "reason": ...}` instead of raising.

HARD RULE (SPEC "Hard rules" #1): never import `livekit` / `livekit.api` /
`livekit.portal` at module import time — only inside the functions that need
them.
"""

from __future__ import annotations

import atexit
import contextlib
import datetime
import ipaddress
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import psutil

from .utils import config as cfg

logger = logging.getLogger(__name__)

# Room / Portal session naming. SPEC "Wire contract": the room is
# "makerlab-<robot_name>", built by BOTH sides from the same raw robot name —
# do not slugify here, or the client's independently-built name stops matching
# and Portal silently never pairs.
ROOM_PREFIX = "makerlab-"

# The port MakerLab's own API/UI listens on (scripts/makerlab.py BACKEND_PORT).
# Repeated rather than imported so this module never pulls in the launcher.
API_PORT = 8000

DEFAULT_LIVEKIT_PORT = 7880

# Where a `brew install livekit` lands when /opt/homebrew/bin is not on the
# PATH of the process that started MakerLab (launchd/GUI-spawned sessions).
LIVEKIT_BINARY_FALLBACKS = (
    "/opt/homebrew/bin/livekit-server",
    "/usr/local/bin/livekit-server",
)

# Tunnel interfaces whose addresses LiveKit must not advertise blindly. See
# `_has_tunnel_interfaces`.
_TUNNEL_IFACE_PREFIXES = ("utun", "tun", "tap", "ppp", "ipsec")
# The benchmarking range MaoMaoYun-style VPNs hand out (198.18.0.0/15). A
# client routes it into its own VPN, so an ICE candidate there is a black hole.
_VPN_NETWORK = ipaddress.ip_network("198.18.0.0/15")
# Carrier-grade NAT space — what Tailscale hands out. Reachable and desirable.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# TTL of the minted JWT. Long enough to cover a full record/train session
# without a mid-session reconnect failing on an expired token.
TOKEN_TTL = datetime.timedelta(hours=6)

# How long to wait for livekit-server to actually be listening before calling
# the start a failure.
_STARTUP_TIMEOUT_S = 8.0

_lock = threading.RLock()
_proc: subprocess.Popen | None = None
_log_handle = None
_keys: tuple[str, str] | None = None
_server_mode = False
_disabled_reason: str | None = None
# True when we found something already listening on the LiveKit port and chose
# not to spawn our own (see `start_livekit_server`).
_external_server = False
_atexit_registered = False
_active_room: str | None = None
# Populated by the Robot-side teleoperator (LiveKitTeleoperator) so
# /portal/status can report who is connected without blocking the event loop
# on a LiveKit API round-trip.
_participants: dict = {"operators": [], "active_operator": None, "metrics": None}
# Cheap TTL cache for the "is something listening on :7880" probe, so a status
# poll can't turn into a per-request blocking connect on the event loop.
_port_probe: tuple[float, bool] = (0.0, False)


# ---------------------------------------------------------------------------
# Paths (derived from utils/config.py — never hardcode ~/.cache here)
# ---------------------------------------------------------------------------


def state_dir() -> Path:
    """Directory holding generated livekit config, logs and keys.

    Derived from `utils.config.ROBOTS_PATH` (read through the module so a test
    that redirects the lerobot cache redirects this too) rather than an
    independent `~/.cache/...` literal.
    """
    return Path(cfg.ROBOTS_PATH).parent / "portal"


def _config_path() -> Path:
    return state_dir() / "livekit.yaml"


def _log_path() -> Path:
    return state_dir() / "livekit-server.log"


def _keys_path() -> Path:
    return state_dir() / "livekit_keys.json"


# ---------------------------------------------------------------------------
# Network discovery
# ---------------------------------------------------------------------------


def livekit_port() -> int:
    """LiveKit signalling port (MAKERLAB_LIVEKIT_PORT overrides)."""
    raw = os.environ.get("MAKERLAB_LIVEKIT_PORT")
    if raw:
        with contextlib.suppress(ValueError):
            return int(raw)
    return DEFAULT_LIVEKIT_PORT


def _interface_addresses() -> list[tuple[str, str]]:
    """[(interface_name, ipv4)] for every up interface with an IPv4 address."""
    out: list[tuple[str, str]] = []
    try:
        stats = psutil.net_if_stats()
        for iface, addrs in psutil.net_if_addrs().items():
            info = stats.get(iface)
            if info is not None and not info.isup:
                continue
            for addr in addrs:
                if addr.family == socket.AF_INET and addr.address:
                    out.append((iface, addr.address))
    except Exception as exc:  # pragma: no cover - psutil platform quirks
        logger.debug("interface enumeration failed: %s", exc)
    return out


def _default_route_ip() -> str | None:
    """The source IP the OS would use to reach the internet, or None.

    A connect() on a UDP socket sends no packets — it only asks the routing
    table which local address would be used — so this works offline-ish and
    never blocks meaningfully.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _is_usable_lan_ip(ip: str) -> bool:
    """True for an address another machine on the LAN (or Tailscale) can use."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return False
    if addr in _VPN_NETWORK:
        return False
    # RFC1918 plus the 100.64/10 CGNAT range Tailscale uses — a Tailscale
    # address is a perfectly good way for the client to reach this station
    # (the RUNBOOK's working config allowlists exactly LAN + Tailscale).
    # Python's `is_private` does NOT cover 100.64/10, hence the explicit test.
    return addr.is_private or addr in _CGNAT_NETWORK


def detect_lan_ips() -> list[str]:
    """Addresses a MakerLab client could point at, best candidate first.

    Never hardcoded: enumerated from the live interface list, with the OS's
    default-route source address promoted to the front when it is usable.
    """
    seen: list[str] = []
    for _iface, ip in _interface_addresses():
        if _is_usable_lan_ip(ip) and ip not in seen:
            seen.append(ip)
    primary = _default_route_ip()
    if primary and _is_usable_lan_ip(primary):
        if primary in seen:
            seen.remove(primary)
        seen.insert(0, primary)
    return seen


def primary_lan_ip(ips: list[str] | None = None) -> str:
    """The single address to advertise. Falls back to loopback if isolated."""
    candidates = detect_lan_ips() if ips is None else ips
    return candidates[0] if candidates else "127.0.0.1"


def _has_tunnel_interfaces() -> bool:
    """True when a VPN tunnel could poison LiveKit's ICE candidate list.

    LiveKit advertises every local address it finds. A MaoMaoYun-style VPN
    (198.18.0.0/15) or another utun/ppp tunnel produces candidates the client
    routes into *its own* VPN — a black hole that manifests as "connected but
    no media". Only in that case do we pin `rtc.ips.includes`; pinning it
    unnecessarily is worse (a wrong allowlist silently kills all connectivity).
    """
    for iface, ip in _interface_addresses():
        if iface.startswith(_TUNNEL_IFACE_PREFIXES):
            return True
        with contextlib.suppress(ValueError):
            if ipaddress.ip_address(ip) in _VPN_NETWORK:
                return True
    return False


def _port_is_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# API key / secret
# ---------------------------------------------------------------------------


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, never leaving it world-readable.

    Both files this is used for carry the LiveKit API secret. Writing first and
    chmod'ing after leaves a window in which the file exists as 0644 under the
    default 022 umask — any local account can read the secret in that window —
    so the temp file is CREATED 0600 by ``os.open`` (the kernel applies the
    mode at creation, and a umask can only take bits away, never add them) and
    only then moved into place. ``os.replace`` keeps the temp file's mode, so
    the destination is 0600 from its very first instant.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # A crashed earlier write would leave the temp file behind and O_EXCL would
    # then fail for good; its content is ours to discard either way.
    with contextlib.suppress(OSError):
        os.unlink(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _tighten_permissions(path: Path) -> None:
    """chmod an existing secret-bearing file back to 0600, loudly. Never raises.

    Covers a file written by an older MakerLab (which chmod'ed after writing,
    and skipped the repair entirely on the load path) or one relaxed by hand.
    """
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return
    if not mode & 0o077:
        return
    logger.warning(
        "Portal: %s was readable by other accounts (mode %o) — tightening it to 0600. "
        "It carries this station's LiveKit API secret; rotate the pair if the machine is shared.",
        path,
        mode,
    )
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _generate_keys() -> tuple[str, str]:
    import secrets

    # 43 url-safe chars: comfortably over LiveKit's 32-char minimum, which is
    # only waived in development mode.
    return f"makerlab{secrets.token_hex(4)}", secrets.token_urlsafe(32)


def _load_keys() -> tuple[str, str]:
    """(api_key, api_secret) for this station, creating them on first use.

    Resolution order:
      1. env (MAKERLAB_LIVEKIT_API_KEY/SECRET, then the LIVEKIT_* names the
         portal examples already use) — lets an operator match an existing
         livekit deployment,
      2. the persisted pair under the cache dir,
      3. a freshly generated pair, persisted 0600.
    If persisting fails we fall back to LiveKit's dev key/secret so the pair
    stays stable across restarts instead of silently rotating (a rotated
    secret would invalidate every token a running client still holds).
    The secret is never returned by any endpoint and never logged.
    """
    global _keys
    with _lock:
        if _keys is not None:
            return _keys
        for key_var, secret_var in (
            ("MAKERLAB_LIVEKIT_API_KEY", "MAKERLAB_LIVEKIT_API_SECRET"),
            ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"),
        ):
            key, secret = os.environ.get(key_var), os.environ.get(secret_var)
            if key and secret:
                logger.info("Portal: using LiveKit API key from %s", key_var)
                _keys = (key, secret)
                return _keys

        path = _keys_path()
        try:
            stored = json.loads(path.read_text())
            key, secret = stored.get("api_key"), stored.get("api_secret")
            if key and secret:
                # Repair the permissions of a pair written before this file
                # created its secrets 0600 (or relaxed since).
                _tighten_permissions(path)
                _keys = (str(key), str(secret))
                return _keys
        except (OSError, ValueError):
            pass

        key, secret = _generate_keys()
        try:
            _write_private(path, json.dumps({"api_key": key, "api_secret": secret}, indent=2))
            logger.info("Portal: generated a LiveKit API key pair at %s", path)
        except OSError as exc:
            logger.warning(
                "Portal: could not persist LiveKit keys at %s (%s) — falling back to "
                "the dev key/secret so tokens stay valid across restarts.",
                path,
                exc,
            )
            key, secret = "devkey", "secret"
        _keys = (key, secret)
        return _keys


# ---------------------------------------------------------------------------
# livekit-server subprocess
# ---------------------------------------------------------------------------


def find_livekit_binary() -> str | None:
    """Path to `livekit-server`, or None if it isn't installed."""
    override = os.environ.get("MAKERLAB_LIVEKIT_BINARY")
    if override:
        return override if os.path.isfile(override) else None
    found = shutil.which("livekit-server")
    if found:
        return found
    return next((p for p in LIVEKIT_BINARY_FALLBACKS if os.path.isfile(p)), None)


def _render_config(key: str, secret: str) -> str:
    """The livekit-server YAML for this station.

    Two gotchas encoded here (both cost real debugging time — see
    portal/examples/python/so101/RUNBOOK.local.md §A2):

    * `bind_addresses` does NOT inherit the `--bind` CLI default once a config
      file is used, so without it livekit listens on localhost only and the
      client's connection is refused from the LAN.
    * `rtc.ips.includes` entries MUST be CIDR (`x.x.x.x/32`). A bare IP makes
      livekit-server exit with "invalid CIDR address" *after* logging
      "starting in development mode", which reads exactly like a good start.

    Hand-rendered rather than yaml.dump'd: no new dependency, and the file
    stays readable for an operator debugging on the station.
    """
    port = livekit_port()
    lines = [
        "# Generated by MakerLab (makerlab/portal_host.py). Regenerated on every",
        "# `makerlab --server` start — edit MAKERLAB_LIVEKIT_* env vars instead.",
        f"port: {port}",
        "bind_addresses:",
        # REQUIRED: a config file drops the --bind CLI default.
        "  - 0.0.0.0",
        "keys:",
        f"  {key}: {secret}",
        "rtc:",
        f"  tcp_port: {port + 1}",
        f"  udp_port: {port + 2}",
        "  use_external_ip: false",
    ]
    if _has_tunnel_interfaces():
        allowed = detect_lan_ips()
        if allowed:
            lines += [
                "  # A VPN/tunnel interface is present, so pin the advertised ICE",
                "  # candidates to this station's real LAN/Tailscale addresses.",
                "  # CIDR form is mandatory — a bare IP makes livekit-server exit.",
                "  ips:",
                "    includes:",
            ]
            lines += [f"      - {ip}/32" for ip in allowed]
    return "\n".join(lines) + "\n"


def _log_tail(limit: int = 12) -> str:
    try:
        lines = [ln.strip() for ln in _log_path().read_text().splitlines() if ln.strip()]
    except OSError:
        return ""
    return " | ".join(lines[-limit:])


def start_livekit_server() -> dict:
    """Start (or adopt) the LiveKit server for this station.

    Returns {"started": bool, "reason": str|None, "url": str, "log_file": str}.
    Never raises: a missing binary or a server that dies on startup disables
    Portal and leaves the rest of MakerLab untouched.
    """
    global _proc, _log_handle, _disabled_reason, _external_server, _atexit_registered
    with _lock:
        port = livekit_port()
        url = livekit_url()
        log_file = str(_log_path())

        if _proc is not None and _proc.poll() is None:
            return {"started": True, "reason": None, "url": url, "log_file": log_file}

        binary = find_livekit_binary()
        if binary is None:
            _disabled_reason = (
                "livekit-server is not installed (not on PATH, not in /opt/homebrew/bin). "
                "Install it with `brew install livekit` — MakerLab runs normally without it, "
                "but a remote client cannot connect."
            )
            logger.warning("Portal disabled: %s", _disabled_reason)
            return {"started": False, "reason": _disabled_reason, "url": url, "log_file": log_file}

        if _port_is_open(port):
            _external_server = True
            _disabled_reason = None
            logger.warning(
                "Portal: something is already listening on :%d — adopting it instead of "
                "starting our own. Its API key/secret must match MakerLab's, otherwise set "
                "MAKERLAB_LIVEKIT_API_KEY / MAKERLAB_LIVEKIT_API_SECRET to the same pair.",
                port,
            )
            return {"started": True, "reason": None, "url": url, "log_file": log_file, "external": True}

        key, secret = _load_keys()
        try:
            state_dir().mkdir(parents=True, exist_ok=True)
            # 0600 from the instant it exists — the rendered config embeds the
            # API secret in its `keys:` block.
            _write_private(_config_path(), _render_config(key, secret))
            _log_handle = open(_log_path(), "a", buffering=1)  # noqa: SIM115 - lives as long as the child
        except OSError as exc:
            _disabled_reason = f"could not write the livekit-server config/log under {state_dir()}: {exc}"
            logger.warning("Portal disabled: %s", _disabled_reason)
            return {"started": False, "reason": _disabled_reason, "url": url, "log_file": log_file}

        try:
            # --dev keeps the RUNBOOK-proven invocation (relaxed TLS/secret
            # checks for a LAN station). It only substitutes devkey/secret when
            # the config carries no keys, so ours survive.
            _proc = subprocess.Popen(
                [binary, "--dev", "--config", str(_config_path())],
                stdout=_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            _disabled_reason = f"could not launch {binary}: {exc}"
            logger.warning("Portal disabled: %s", _disabled_reason)
            return {"started": False, "reason": _disabled_reason, "url": url, "log_file": log_file}

        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            # Both checks matter: livekit-server logs "starting in development
            # mode" and THEN exits on a bad config, so a log line (or even a
            # brief listen) is not proof it is up.
            if _proc.poll() is not None:
                _disabled_reason = (
                    f"livekit-server exited immediately (code {_proc.returncode}). "
                    f"Last log lines: {_log_tail()}"
                )
                logger.error("Portal disabled: %s", _disabled_reason)
                _proc = None
                return {"started": False, "reason": _disabled_reason, "url": url, "log_file": log_file}
            if _port_is_open(port):
                break
            time.sleep(0.2)
        else:
            _disabled_reason = f"livekit-server did not listen on :{port} within {_STARTUP_TIMEOUT_S:.0f}s"
            logger.error("Portal disabled: %s", _disabled_reason)
            stop_livekit_server()
            return {"started": False, "reason": _disabled_reason, "url": url, "log_file": log_file}

        _disabled_reason = None
        _external_server = False
        if not _atexit_registered:
            atexit.register(stop_livekit_server)
            _atexit_registered = True
        logger.info("Portal: livekit-server running on %s (pid %d, log %s)", url, _proc.pid, log_file)
        return {"started": True, "reason": None, "url": url, "log_file": log_file}


def stop_livekit_server(timeout: float = 5.0) -> None:
    """Terminate our livekit-server child (idempotent, never raises)."""
    global _proc, _log_handle
    with _lock:
        proc, _proc = _proc, None
        handle, _log_handle = _log_handle, None
    if proc is not None and proc.poll() is None:
        logger.info("Portal: stopping livekit-server (pid %d)...", proc.pid)
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=timeout)
    if handle is not None:
        with contextlib.suppress(OSError):
            handle.close()


def livekit_running() -> bool:
    """True when a LiveKit server is up for this station."""
    global _port_probe
    with _lock:
        if _proc is not None:
            return _proc.poll() is None
        if not _external_server:
            return False
    # Adopted external server: probe, but at most every 2s so a status poll
    # can't turn into a per-request blocking connect on the event loop.
    now = time.monotonic()
    checked_at, was_open = _port_probe
    if now - checked_at < 2.0:
        return was_open
    is_open = _port_is_open(livekit_port())
    _port_probe = (now, is_open)
    return is_open


# ---------------------------------------------------------------------------
# Server-mode lifecycle (driven by `makerlab --server`)
# ---------------------------------------------------------------------------


def set_server_mode(enabled: bool) -> None:
    """Mark this process as a Portal host. Called by the launcher only."""
    global _server_mode
    _server_mode = bool(enabled)


def server_mode() -> bool:
    return _server_mode


def set_active_room(room: str | None) -> None:
    """Record the room the Robot side is currently serving (for /portal/*)."""
    global _active_room
    _active_room = room or None


def report_participants(
    operators: list | None = None,
    active_operator: str | None = None,
    metrics: dict | None = None,
) -> None:
    """Push connected-operator state for /portal/status.

    The Robot-side teleoperator owns this knowledge; querying the LiveKit API
    from the HTTP handler would mean an async round-trip on the event loop.
    """
    _participants["operators"] = list(operators or [])
    _participants["active_operator"] = active_operator
    _participants["metrics"] = metrics


# ---------------------------------------------------------------------------
# HTTP handlers (called by server.py; must not block the event loop)
# ---------------------------------------------------------------------------


def _livekit_api_available() -> bool:
    """True when the `livekit` Python SDK is importable — without importing it."""
    try:
        import importlib.util

        return importlib.util.find_spec("livekit.api") is not None
    except (ImportError, ValueError):
        return False


def livekit_url(host: str | None = None) -> str:
    """The ws:// URL a client should dial (MAKERLAB_LIVEKIT_URL overrides)."""
    override = os.environ.get("MAKERLAB_LIVEKIT_URL")
    if override:
        return override
    return f"ws://{host or primary_lan_ip()}:{livekit_port()}"


def api_reachable_at(host: str | None = None) -> str:
    return f"http://{host or primary_lan_ip()}:{API_PORT}"


def room_name(robot_name: str) -> str:
    """SPEC wire contract: the Portal room is `makerlab-<robot_name>`.

    Concatenated raw (no slugging) because the client builds the same string
    from the same robot name — any normalisation here would have to be
    mirrored exactly there, and a mismatch shows up as "the arm never moves".
    """
    return f"{ROOM_PREFIX}{robot_name}"


def _resolve_room() -> str:
    """Best guess at the room this station serves, for /portal/info."""
    if _active_room:
        return _active_room
    try:
        records = cfg.list_robot_records()
    except Exception:  # pragma: no cover - defensive: never break /portal/info
        return ""
    if len(records) == 1:
        return room_name(records[0].get("name", ""))
    return ""


def _enabled_state() -> tuple[bool, str | None]:
    """(enabled, reason) — why Portal hosting is or isn't available here."""
    if not _livekit_api_available():
        return False, (
            "the `livekit` Python SDK is not installed in this environment, so MakerLab "
            "cannot mint Portal tokens. Install the portal extras on this machine."
        )
    if find_livekit_binary() is None and not livekit_running():
        return False, (
            "livekit-server is not installed (not on PATH, not in /opt/homebrew/bin). "
            "Install it with `brew install livekit`."
        )
    if _disabled_reason:
        return False, _disabled_reason
    return True, None


def handle_portal_info(client_host: str | None = None) -> dict:
    """GET /portal/info — can this machine host Portal, and where to dial it.

    `client_host` is the hostname the caller actually used to reach this API
    (from the request's Host header). It matters: this station may be reachable
    on a LAN address, a Tailscale name, or both, and `primary_lan_ip()` only
    ever knows the LAN one. Handing an off-LAN client a LAN ws:// URL produces
    a Portal connect that simply times out with nothing in the logs to explain
    it. Whatever host reached the API is, by construction, a host that can
    reach this machine — so reuse it and only swap the port.
    """
    enabled, reason = _enabled_state()
    host = client_host or primary_lan_ip()
    info = {
        "enabled": enabled,
        "livekit_url": livekit_url(host),
        "room": _resolve_room(),
        "room_prefix": ROOM_PREFIX,
        "api_reachable_at": api_reachable_at(host),
        "server_mode": server_mode(),
        "running": livekit_running(),
        "lan_ips": detect_lan_ips(),
    }
    if not enabled:
        info["reason"] = reason
    return info


def handle_portal_status() -> dict:
    """GET /portal/status — live state of the LiveKit server and its operators."""
    status = {
        "livekit_running": livekit_running(),
        "operators": list(_participants["operators"]),
        "active_operator": _participants["active_operator"],
        "server_mode": server_mode(),
        "log_file": str(_log_path()),
    }
    if _participants["metrics"] is not None:
        status["metrics"] = _participants["metrics"]
    if _disabled_reason:
        status["reason"] = _disabled_reason
    return status


def handle_portal_token(identity: str, room: str, client_host: str | None = None) -> dict:
    """POST /portal/token — mint a LiveKit JWT for a remote operator.

    The API secret stays on this machine: only the signed token goes out.
    Grants match the SPEC wire contract — `can_update_own_metadata` is NOT
    optional, both Portal roles self-set `lk.portal.role` on connect and the
    connection fails outright without it.

    `client_host` — see handle_portal_info(). The returned `url` is what the
    client will actually dial, so it must be derived from the host that reached
    us, not from the LAN IP; otherwise an off-LAN operator (e.g. over Tailscale)
    receives a valid token pointing at an unreachable address.
    """
    identity = (identity or "").strip()
    room = (room or "").strip()
    if not identity or not room:
        return {
            "success": False,
            "status_code": 400,
            "message": "Both `identity` and `room` are required.",
        }
    if len(identity) > 256 or len(room) > 256:
        return {"success": False, "status_code": 400, "message": "`identity`/`room` are too long."}

    if not livekit_running():
        return {
            "success": False,
            "status_code": 503,
            "message": (
                "No LiveKit server is running on this machine. Start MakerLab with "
                "`makerlab --server` on the robot station."
            ),
        }

    try:
        from livekit import api
        from livekit.protocol.room import RoomConfiguration
    except ImportError as exc:
        return {
            "success": False,
            "status_code": 503,
            "message": f"The `livekit` Python SDK is not installed on the server: {exc}",
        }

    key, secret = _load_keys()
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_update_own_metadata=True,
    )
    # min/max playout delay pin the receiver's jitter buffer to "as live as
    # possible" — teleop is unusable with the default buffering.
    room_cfg = RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1)
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_room_config(room_cfg)
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )
    logger.info("Portal: minted a token for identity=%s room=%s", identity, room)
    return {"token": token, "url": livekit_url(client_host), "room": room}
