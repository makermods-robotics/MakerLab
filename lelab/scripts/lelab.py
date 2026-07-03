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

"""
LeLab launcher.

Default mode: starts the FastAPI backend on :8000, which serves the
pre-built frontend at /. Opens the user's browser to the local app.

--dev mode: spawns the Vite dev server (frontend/, port 8080) for HMR
and starts uvicorn with --reload. Opens the browser to :8080.
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_PATH = PROJECT_ROOT / "frontend"
FRONTEND_DIST = FRONTEND_PATH / "dist"
BACKEND_PORT = 8000
FRONTEND_DEV_PORT = 8080
ENTRY_POINT_NAMES = ("lelab", "lelab-station", "makerlabs")


def _ensure_path_symlinks(source_dir: Path | None = None, bin_dir: Path | None = None) -> None:
    """Self-install the entry points onto PATH (idempotent, best-effort).

    pip has no post-install hook, so the first run by full path does the
    INSTALL.md symlink step itself: each venv entry point gets a symlink in
    ~/.local/bin. Correct links are left alone; stale symlinks (an old
    clone's venv) are repointed; anything that is NOT a symlink is never
    clobbered. Failures only log — PATH convenience must never block a
    server start. Set LELAB_NO_PATH_LINK=1 to opt out.
    """
    if os.name != "posix" or os.environ.get("LELAB_NO_PATH_LINK"):
        return
    try:
        source_dir = source_dir or Path(sys.executable).parent
        bin_dir = bin_dir or Path.home() / ".local" / "bin"
        created: list[str] = []
        for name in ENTRY_POINT_NAMES:
            source = source_dir / name
            if not source.is_file():
                continue  # partial env (entry point not installed here)
            link = bin_dir / name
            if link.is_symlink():
                if link.resolve() == source.resolve():
                    continue
                link.unlink()  # stale: points into an old venv/clone
            elif link.exists():
                logger.warning(
                    "Not shadowing %s — it exists and is not a symlink; remove it "
                    "manually if `%s` should run this venv's copy.",
                    link,
                    name,
                )
                continue
            bin_dir.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source)
            created.append(name)
        if created:
            logger.info(
                "🔗 Linked %s into %s — new shells can run them from any directory",
                ", ".join(created),
                bin_dir,
            )
            if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
                logger.warning("%s is not on your PATH — add it in your shell profile", bin_dir)
    except Exception as exc:
        logger.debug("PATH symlink self-install skipped: %s", exc)


def _wait_for_port(port: int, timeout: int = 30) -> bool:
    for _ in range(timeout):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            return True
        time.sleep(1)
    return False


def _open_browser_when_ready():
    """Background-thread helper: poll the port, open the browser when up."""
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.5)
            continue
        logger.info("🌐 Opening browser...")
        webbrowser.open(f"http://localhost:{BACKEND_PORT}/")
        return


def _run_prod(lan: bool = False):
    """Serve built frontend from backend on a single port.

    `lan` binds 0.0.0.0 for headless stations serving other machines on the
    network; it also skips the open-a-local-browser step (there is no local
    browser worth opening in that deployment).
    """
    if not FRONTEND_DIST.exists():
        logger.error(f"❌ Built frontend not found at {FRONTEND_DIST}")
        logger.error("   Run `npm run build` in frontend/ first, or use `lelab --dev`.")
        sys.exit(1)

    host = "0.0.0.0" if lan else "127.0.0.1"  # noqa: S104
    if lan:
        logger.info("🚀 Starting LeLab on http://0.0.0.0:%d (LAN) ...", BACKEND_PORT)
    else:
        logger.info("🚀 Starting LeLab on http://localhost:%d ...", BACKEND_PORT)
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    # Run uvicorn in the main thread so its native SIGINT handler works,
    # and bound graceful shutdown so a stuck WebSocket can't hang Ctrl+C.
    uvicorn.run(
        "lelab.server:app",
        host=host,
        port=BACKEND_PORT,
        log_level="info",
        reload=False,
        timeout_graceful_shutdown=2,
    )


def _run_dev():
    """Vite dev server (HMR) + uvicorn --reload."""
    if not FRONTEND_PATH.exists():
        logger.error(f"❌ Frontend not found at {FRONTEND_PATH}")
        sys.exit(1)

    logger.info("📦 Installing frontend deps...")
    subprocess.run(["npm", "install"], check=True, cwd=FRONTEND_PATH)

    logger.info("🎨 Starting Vite dev server (port %d)...", FRONTEND_DEV_PORT)
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_PATH,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    if not _wait_for_port(FRONTEND_DEV_PORT):
        logger.error("❌ Frontend never came up")
        frontend_process.terminate()
        sys.exit(1)

    logger.info("🚀 Starting backend (port %d) with --reload...", BACKEND_PORT)
    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "lelab.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(BACKEND_PORT),
            "--reload",
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        start_new_session=True,
    )

    if not _wait_for_port(BACKEND_PORT, timeout=15):
        logger.error("❌ Backend never came up")
        for p in (backend_process, frontend_process):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
        sys.exit(1)

    logger.info("🌐 Opening browser...")
    webbrowser.open(f"http://localhost:{FRONTEND_DEV_PORT}/")

    logger.info("✅ Dev mode running — Ctrl+C to stop")
    logger.info("   Frontend: http://localhost:%d", FRONTEND_DEV_PORT)
    logger.info("   Backend:  http://localhost:%d", BACKEND_PORT)

    def shutdown(signum, frame):
        logger.info("🛑 Shutting down...")
        for name, p in [("backend", backend_process), ("frontend", frontend_process)]:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
            except Exception:
                pass
            logger.info(f"  ✅ {name} stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(2)
        if backend_process.poll() is not None:
            logger.error("❌ Backend died")
            shutdown(None, None)
        if frontend_process.poll() is not None:
            logger.error("❌ Frontend died")
            shutdown(None, None)


def main():
    parser = argparse.ArgumentParser(prog="lelab", description="Run LeLab")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: Vite HMR + uvicorn --reload (requires Node.js)",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Headless station mode: bind 0.0.0.0 (serve other machines), don't open a browser",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1: every Hub call fails fast (all hardware flows work offline)",
    )
    args = parser.parse_args()

    _ensure_path_symlinks()

    if args.offline:
        # Must land in the environment before lelab.server (and its
        # huggingface_hub import) loads — uvicorn imports the app lazily, so
        # setting it here covers both prod and the dev subprocess (env copy).
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.dev:
        if args.lan:
            logger.warning("--lan is ignored in --dev mode (Vite serves localhost only)")
        _run_dev()
    else:
        _run_prod(lan=args.lan)


def station():
    """Entry point for headless robot stations: `lelab --lan --offline`.

    Installed as `lelab-station` (see pyproject.toml) so the posture is a
    first-class command — and what deploy/lelab-station.service runs at boot.
    Extra CLI args still pass through.
    """
    sys.argv = [sys.argv[0], "--lan", "--offline", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    main()
