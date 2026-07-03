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


def test_localize_allows_cloud_resume_from_hub() -> None:
    """A cloud resume signals via resume_from_hub_repo (the wrapper downloads the
    checkpoint from the Hub), not a host-local config_path — so localization must
    NOT reject it. The host config_path stays unset here; the runner sets the
    container path later."""
    from lelab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_device == "cuda"


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from lelab.runners.hf_cloud import localize_config_for_cloud

    local = _request(policy_pretrained_path="/host/checkpoints/000500/pretrained_model")
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        localize_config_for_cloud(local, "t4-small")

    hub = _request(policy_pretrained_path="user/some-model")
    localize_config_for_cloud(hub, "t4-small")  # no raise
    assert hub.policy_pretrained_path == "user/some-model"


# -- in-container installer ladder (the image's uv venv ships no pip) --


def test_install_plan_prefers_uv() -> None:
    """The lerobot-gpu image's venv is uv-created (no pip module), so uv on
    PATH must win, with --python pinning the install into this interpreter."""
    from lelab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("lerobot @ url", "/venv/bin/python", "/usr/local/bin/uv", True, True)
    assert label == "uv"
    assert cmds == [
        [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--python",
            "/venv/bin/python",
            "--no-cache",
            "lerobot @ url",
        ]
    ]


def test_install_plan_falls_back_to_pip_without_uv() -> None:
    from lelab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, True, True)
    assert label == "pip"
    assert cmds == [["/py", "-m", "pip", "install", "--no-cache-dir", "spec"]]


def test_install_plan_bootstraps_pip_via_ensurepip_as_last_resort() -> None:
    from lelab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, False, True)
    assert label == "ensurepip+pip"
    assert cmds == [
        ["/py", "-m", "ensurepip", "--upgrade"],
        ["/py", "-m", "pip", "install", "--no-cache-dir", "spec"],
    ]


def test_install_plan_reports_no_installer() -> None:
    from lelab.runners.hf_cloud import _install_plan

    assert _install_plan("spec", "/py", None, False, False) == (None, [])


# -- wrapper sanity --


def test_wrapper_source_compiles_and_launches_an_argv_list() -> None:
    """The wrapper must pass the trainer argv to Popen as a LIST (splitting a
    joined string was the bug-3 hypothesis — it is not the case and must stay
    that way) and quote its log line so spaced values read unambiguously."""
    from lelab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # syntactically valid
    assert "subprocess.Popen(list(trainer_argv)" in WRAPPER_SOURCE
    assert "shlex.join(trainer_argv)" in WRAPPER_SOURCE
    assert re.search(r"^import .*\bshlex\b", WRAPPER_SOURCE, re.MULTILINE)  # imported up top


def test_wrapper_source_handles_resume_download() -> None:
    """Cloud resume: the wrapper must parse --resume-from, download the parent
    checkpoint tree, refuse when training_state/ is absent, and pre-seed `seen`
    so it never re-uploads the checkpoint it just pulled down."""
    from lelab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # still valid with the resume block
    assert "--resume-from=" in WRAPPER_SOURCE
    assert "snapshot_download" in WRAPPER_SOURCE
    assert "training_state" in WRAPPER_SOURCE
    assert "seen.add(step_dir)" in WRAPPER_SOURCE


def test_cloud_resume_argv_keeps_lineage_in_parent_repo() -> None:
    """A cloud-resume config resolves to a --config_path at the container path and
    pushes into the parent's repo (same lineage), with resume essentials only."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/ds",
        resume=True,
        steps=20000,
        policy_push_to_hub=True,
        policy_repo_id="user/parent-run",
        config_path="/tmp/lelab/train/checkpoints/005000/pretrained_model/train_config.json",
    )
    cmd = build_training_command(req, output_dir="/tmp/lelab/train")
    assert "--config_path=/tmp/lelab/train/checkpoints/005000/pretrained_model/train_config.json" in cmd
    assert cmd[cmd.index("--policy.push_to_hub") + 1] == "true"
    assert cmd[cmd.index("--policy.repo_id") + 1] == "user/parent-run"
    assert cmd[cmd.index("--resume") + 1] == "true"
    # Inherited from the checkpoint — never re-passed on resume.
    assert "--dataset.repo_id" not in cmd
    assert "--policy.type" not in cmd


def test_wrapper_source_inlines_the_tested_install_plan() -> None:
    """The wrapper's installer choice is _install_plan's source inlined
    verbatim, so the in-container code is exactly what the unit tests above
    exercised — uv first (shutil.which), pip / ensurepip as fallbacks."""
    import inspect

    from lelab.runners.hf_cloud import WRAPPER_SOURCE, _install_plan

    assert inspect.getsource(_install_plan) in WRAPPER_SOURCE
    assert "__INSTALL_PLAN_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert 'shutil.which("uv")' in WRAPPER_SOURCE
    assert "no uv, pip, or ensurepip" in WRAPPER_SOURCE  # clear terminal message
