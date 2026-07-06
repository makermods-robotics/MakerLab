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
"""Hardware-side fakes used across the test suite."""

from __future__ import annotations

from typing import Any


class FakeRobot:
    """Stand-in for a connected SO-101 follower.

    Records every method call on `self.calls` so tests can assert on it.
    Methods are deliberately the minimum surface MakerLab actually invokes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._connected = False
        self.init_args = args
        self.init_kwargs = kwargs

    def connect(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("connect", args, kwargs))
        self._connected = True

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("disconnect", args, kwargs))
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> dict[str, float]:
        self.calls.append(("get_observation", (), {}))
        return {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls.append(("send_action", (action,), {}))
        return action


class FakeTeleoperator(FakeRobot):
    """Stand-in for a connected SO-101 leader. Same surface as FakeRobot."""

    def get_action(self) -> dict[str, float]:
        self.calls.append(("get_action", (), {}))
        return {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0}


def patch_so101_configs(monkeypatch) -> None:
    """Swap SO101 config constructors so any code path constructing them
    gets a no-op factory that returns a plain dataclass-like stub.

    Use sparingly — most tests prefer to patch the higher-level entry points
    (e.g. `lerobot.record.record`) instead of the config classes.
    """

    class _StubConfig:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    monkeypatch.setattr("lerobot.robots.so101_follower.SO101FollowerConfig", _StubConfig, raising=False)
    monkeypatch.setattr("lerobot.teleoperators.so101_leader.SO101LeaderConfig", _StubConfig, raising=False)
