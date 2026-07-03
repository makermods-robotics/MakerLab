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
"""Tests for lelab.runners.hf_cloud — covers the host-side wandb credential
resolution, the pinned-lerobot spec derivation, and the cloud-boundary config
localization. HfCloudJobRunner itself talks to HF Jobs and is not unit-
testable without a heavy mock of HfApi; we intentionally leave it for
integration tests."""

from __future__ import annotations

import netrc
import re
import tomllib
from pathlib import Path

import pytest


def test_resolve_wandb_api_key_prefers_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.setenv("WANDB_API_KEY", "env-key-123")
    assert resolve_wandb_api_key() == "env-key-123"


def test_resolve_wandb_api_key_falls_back_to_netrc(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WANDB_API_KEY is unset, the function must read the same place
    `wandb login` writes — ~/.netrc under machine api.wandb.ai."""
    from lelab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            assert host == "api.wandb.ai"
            return ("login", "account", "netrc-key-456")

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() == "netrc-key-456"


def test_resolve_wandb_api_key_returns_none_when_netrc_has_no_wandb_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return None

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var, no ~/.netrc — neither source has it, caller decides."""
    from lelab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_missing():
        raise FileNotFoundError("~/.netrc")

    monkeypatch.setattr(netrc, "netrc", _raise_missing)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_parse():
        raise netrc.NetrcParseError("bad netrc", "~/.netrc", 1)

    monkeypatch.setattr(netrc, "netrc", _raise_parse)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_password_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty password from netrc is treated as missing — the helper
    contract is 'returns the usable key or None', not 'returns whatever
    netrc happened to have'."""
    from lelab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return ("login", "account", "")

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() is None


# -- pinned-lerobot spec derivation (version-skew fix) --


def _pyproject_lerobot_pin() -> str:
    """The raw lerobot dependency line from this repo's pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    return next(d for d in deps if d.startswith("lerobot"))


def _spec_extras(spec: str) -> set[str]:
    m = re.match(r"lerobot\[(?P<extras>[^\]]*)\]", spec)
    assert m, f"no extras block in {spec!r}"
    return set(m.group("extras").split(","))


def test_cloud_lerobot_spec_carries_the_pyproject_pinned_ref() -> None:
    """The container install spec must reference the exact ref pinned in
    pyproject.toml — never a hardcoded second copy, never :latest."""
    from lelab.runners.hf_cloud import cloud_lerobot_spec

    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]  # the sha at the end of git+https://…@<sha>
    spec = cloud_lerobot_spec("act")
    assert ref in spec
    assert "latest" not in spec


def test_cloud_lerobot_spec_uses_archive_tarball_not_git() -> None:
    """A GitHub git+ pin is rewritten to the source archive tarball so pip in
    the container can install it without a git binary."""
    from lelab.runners.hf_cloud import cloud_lerobot_spec

    spec = cloud_lerobot_spec("act")
    assert "git+" not in spec
    assert re.search(r"@ https://github\.com/.+/archive/[0-9a-f]+\.tar\.gz$", spec)


def test_cloud_lerobot_spec_drops_host_only_extras_and_adds_policy_extra() -> None:
    from lelab.runners.hf_cloud import cloud_lerobot_spec

    act = _spec_extras(cloud_lerobot_spec("act"))
    assert "feetech" not in act  # serial motor bus: host-only
    assert "training" in act
    assert "core_scripts" in act  # provides lerobot_train

    smolvla = _spec_extras(cloud_lerobot_spec("smolvla"))
    assert smolvla == act | {"smolvla"}

    pi0_fast = _spec_extras(cloud_lerobot_spec("pi0_fast"))
    assert pi0_fast == act | {"pi"}


def test_cloud_lerobot_spec_falls_back_to_pyproject_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running from a source tree without installed LeLab metadata must still
    derive the pin — from pyproject.toml directly."""
    from lelab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "requires", lambda name: None)
    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]
    assert ref in hf_cloud.cloud_lerobot_spec("act")


# -- cloud-boundary config localization (device-leak fix) --


def _request(**overrides):
    from lelab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", **overrides)


def test_localize_forces_flavor_device_over_host_detection() -> None:
    """The host's auto-detected device (mps on a Mac) must never reach a cloud
    job: GPU flavors force cuda, cpu tiers force cpu."""
    from lelab.runners.hf_cloud import localize_config_for_cloud
    from lelab.train import build_training_command

    for host_device in ("auto", "mps", "cpu", None):
        config = _request(policy_device=host_device)
        localize_config_for_cloud(config, "t4-small")
        assert config.policy_device == "cuda"
        cmd = build_training_command(config, "/tmp/out")
        assert cmd[cmd.index("--policy.device") + 1] == "cuda"

    cpu_config = _request(policy_device="mps")
    localize_config_for_cloud(cpu_config, "cpu-upgrade")
    assert cpu_config.policy_device == "cpu"


def test_localize_clears_host_local_dataset_root() -> None:
    from lelab.runners.hf_cloud import localize_config_for_cloud
    from lelab.train import build_training_command

    config = _request(dataset_root="/Users/someone/.cache/huggingface/lerobot/user/ds")
    localize_config_for_cloud(config, "t4-small")
    assert config.dataset_root is None
    assert "--dataset.root" not in build_training_command(config, "/tmp/out")


def test_localize_rejects_resume_from_host_checkpoint() -> None:
    from lelab.runners.hf_cloud import localize_config_for_cloud

    config = _request(
        resume=True, config_path="/host/run/checkpoints/5000/pretrained_model/train_config.json"
    )
    with pytest.raises(ValueError, match="[Rr]esum"):
        localize_config_for_cloud(config, "t4-small")


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from lelab.runners.hf_cloud import localize_config_for_cloud

    local = _request(policy_pretrained_path="/host/checkpoints/000500/pretrained_model")
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        localize_config_for_cloud(local, "t4-small")

    hub = _request(policy_pretrained_path="user/some-model")
    localize_config_for_cloud(hub, "t4-small")  # no raise
    assert hub.policy_pretrained_path == "user/some-model"


# -- wrapper sanity --


def test_wrapper_source_compiles_and_launches_an_argv_list() -> None:
    """The wrapper must pass the trainer argv to Popen as a LIST (splitting a
    joined string was the bug-3 hypothesis — it is not the case and must stay
    that way) and quote its log line so spaced values read unambiguously."""
    from lelab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # syntactically valid
    assert "subprocess.Popen(list(trainer_argv)" in WRAPPER_SOURCE
    assert "shlex.join(trainer_argv)" in WRAPPER_SOURCE
    assert re.search(r"shlex", WRAPPER_SOURCE.splitlines()[1])  # imported up top
