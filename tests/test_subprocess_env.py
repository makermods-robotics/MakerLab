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
"""Tests for makerlab.utils.subprocess_env."""

from __future__ import annotations

import subprocess as subprocess_module

import pytest

from makerlab.utils.subprocess_env import process_isolation_kwargs, utf8_child_env


def test_utf8_child_env_forces_pythonioencoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_EXISTING_VAR", "keep-me")
    env = utf8_child_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["SOME_EXISTING_VAR"] == "keep-me"


def test_utf8_child_env_applies_extra_on_top() -> None:
    env = utf8_child_env(PYTHONUNBUFFERED="1")
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_process_isolation_kwargs_uses_new_process_group_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start_new_session=True is a documented no-op on Windows (CPython's
    Windows _execute_child ignores it), so the Windows branch must use
    creationflags=CREATE_NEW_PROCESS_GROUP instead — not start_new_session."""
    monkeypatch.setattr("makerlab.utils.subprocess_env.platform.system", lambda: "Windows")
    kwargs = process_isolation_kwargs()
    # getattr with the same 0x00000200 fallback as the implementation: this
    # test forces the Windows branch to run on whatever OS actually hosts
    # it (including our POSIX CI), where the real attribute doesn't exist.
    expected_flag = getattr(subprocess_module, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    assert kwargs == {"creationflags": expected_flag}
    assert "start_new_session" not in kwargs


def test_process_isolation_kwargs_uses_start_new_session_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("makerlab.utils.subprocess_env.platform.system", lambda: "Linux")
    assert process_isolation_kwargs() == {"start_new_session": True}
