"""Environment helpers for spawning Python child processes."""

from __future__ import annotations

import os
import platform
import subprocess


def utf8_child_env(**extra: str) -> dict[str, str]:
    """Copy the current environment with PYTHONIOENCODING forced to utf-8.

    Needed because a piped/non-console stdout on Windows otherwise falls back
    to the process's ANSI codepage, which can't encode non-ASCII output and
    crashes Python children that print it. Any `**extra` entries (e.g.
    PYTHONUNBUFFERED="1") are applied on top.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def process_isolation_kwargs() -> dict[str, object]:
    """Popen kwargs that keep a spawned child alive through signals sent to
    this process (e.g. a uvicorn --reload restart, or a user hitting Ctrl+C
    on the parent's console).

    start_new_session=True (setsid) is how POSIX does this, but it is
    silently discarded on Windows: CPython's Windows _execute_child names
    the parameter unused_start_new_session and never passes it to
    CreateProcess, so it provides zero isolation there — confirmed by
    reading that source directly, not inferred. The Windows equivalent is
    creationflags=CREATE_NEW_PROCESS_GROUP, which Microsoft's docs state
    makes a process ignore a console-wide Ctrl+C.

    NOTE: unlike the POSIX path, this hasn't been verified by observing a
    live Ctrl+C actually fail to cascade on Windows (that requires a real
    interactive console/session). Re-verify manually on a real Windows
    terminal before relying on it under load.
    """
    if platform.system() == "Windows":
        # getattr, not a direct attribute reference: CREATE_NEW_PROCESS_GROUP
        # only exists on the Windows build of the subprocess module
        # (CPython docs mark it "Availability: Windows"). A direct reference
        # would raise AttributeError under a mocked-platform unit test
        # running on our POSIX CI, even though this branch never actually
        # executes there in production (platform.system() isn't mocked
        # there). 0x00000200 is the documented win32 constant value.
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": creationflags}
    return {"start_new_session": True}
