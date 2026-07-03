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
"""Tests for lelab.scripts.lelab — covers `_wait_for_port` and
`_ensure_path_symlinks`. The launcher's `_run_prod` / `_run_dev` / `main`
functions are CLI/process glue (they call uvicorn.run, spawn npm, install
SIGINT handlers) and have no unit-testable seam without rewriting them; they
are left to manual smoke testing."""

from __future__ import annotations

import socket
import threading

import pytest


def _bind_listener() -> tuple[socket.socket, int]:
    """Bind a real TCP listener on an ephemeral localhost port. Returns the
    socket (caller must close) and its actual port number."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server, server.getsockname()[1]


def test_wait_for_port_returns_true_when_port_is_open() -> None:
    from lelab.scripts.lelab import _wait_for_port

    server, port = _bind_listener()
    try:
        assert _wait_for_port(port, timeout=2) is True
    finally:
        server.close()


def test_wait_for_port_returns_false_when_port_never_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch sleep so we don't actually block for `timeout` seconds — the
    function's whole loop body is fast otherwise."""
    from lelab.scripts.lelab import _wait_for_port

    monkeypatch.setattr("lelab.scripts.lelab.time.sleep", lambda _s: None)
    # Pick an ephemeral port from the OS, then close it so it's not bound.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert _wait_for_port(port, timeout=2) is False


def test_wait_for_port_returns_true_immediately_for_already_open_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check that the success path doesn't sleep at all — guards
    against accidentally adding a leading delay."""
    from lelab.scripts.lelab import _wait_for_port

    sleep_calls = []
    monkeypatch.setattr("lelab.scripts.lelab.time.sleep", lambda s: sleep_calls.append(s))

    server, port = _bind_listener()
    # Drain any incoming connection so the listener stays healthy.
    accept_thread = threading.Thread(target=lambda: server.accept() if server else None, daemon=True)
    accept_thread.start()

    try:
        assert _wait_for_port(port, timeout=5) is True
        assert sleep_calls == []
    finally:
        server.close()


def _fake_entry_points(tmp_path):
    """A fake venv bin dir containing all three entry-point scripts."""
    from lelab.scripts.lelab import ENTRY_POINT_NAMES

    source_dir = tmp_path / "venv-bin"
    source_dir.mkdir()
    for name in ENTRY_POINT_NAMES:
        (source_dir / name).write_text("#!/bin/sh\n")
    return source_dir


def test_ensure_path_symlinks_links_all_entry_points(tmp_path) -> None:
    from lelab.scripts.lelab import ENTRY_POINT_NAMES, _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"  # deliberately absent: must be created

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    for name in ENTRY_POINT_NAMES:
        link = bin_dir / name
        assert link.is_symlink()
        assert link.resolve() == (source_dir / name).resolve()


def test_ensure_path_symlinks_is_idempotent(tmp_path) -> None:
    from lelab.scripts.lelab import ENTRY_POINT_NAMES, _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)
    before = {name: (bin_dir / name).lstat().st_ino for name in ENTRY_POINT_NAMES}
    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    # Correct links are left alone, not unlinked and re-created.
    assert {name: (bin_dir / name).lstat().st_ino for name in ENTRY_POINT_NAMES} == before


def test_ensure_path_symlinks_repoints_stale_link(tmp_path) -> None:
    from lelab.scripts.lelab import _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    old_venv = tmp_path / "old-venv-bin"
    old_venv.mkdir()
    (old_venv / "makerlabs").write_text("#!/bin/sh\n")
    (bin_dir / "makerlabs").symlink_to(old_venv / "makerlabs")

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert (bin_dir / "makerlabs").resolve() == (source_dir / "makerlabs").resolve()


def test_ensure_path_symlinks_never_clobbers_regular_files(tmp_path) -> None:
    from lelab.scripts.lelab import _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    foreign = bin_dir / "lelab"
    foreign.write_text("someone else's script\n")

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert not foreign.is_symlink()
    assert foreign.read_text() == "someone else's script\n"
    # The other two are still linked.
    assert (bin_dir / "makerlabs").is_symlink()
    assert (bin_dir / "lelab-station").is_symlink()


def test_ensure_path_symlinks_skips_missing_entry_points(tmp_path) -> None:
    from lelab.scripts.lelab import _ensure_path_symlinks

    source_dir = tmp_path / "venv-bin"
    source_dir.mkdir()
    (source_dir / "lelab").write_text("#!/bin/sh\n")  # only one installed
    bin_dir = tmp_path / "local-bin"

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert (bin_dir / "lelab").is_symlink()
    assert not (bin_dir / "makerlabs").exists()
    assert not (bin_dir / "lelab-station").exists()


def test_ensure_path_symlinks_env_opt_out(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab.scripts.lelab import _ensure_path_symlinks

    monkeypatch.setenv("LELAB_NO_PATH_LINK", "1")
    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert not bin_dir.exists()
