"""Environment helpers for spawning Python child processes."""

from __future__ import annotations

import os


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
