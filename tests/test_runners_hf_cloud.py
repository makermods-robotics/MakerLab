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
"""Tests for makermodslab.runners.hf_cloud — covers the host-side wandb credential
resolution, the pinned-lerobot spec derivation, and the cloud-boundary config
localization. HfCloudJobRunner itself talks to HF Jobs and is not unit-
testable without a heavy mock of HfApi; we intentionally leave it for
integration tests."""

from __future__ import annotations

import json
import netrc
import re
import tomllib
from pathlib import Path

import pytest


def test_resolve_wandb_api_key_prefers_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.setenv("WANDB_API_KEY", "env-key-123")
    assert resolve_wandb_api_key() == "env-key-123"


def test_resolve_wandb_api_key_falls_back_to_netrc(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WANDB_API_KEY is unset, the function must read the same place
    `wandb login` writes — ~/.netrc under machine api.wandb.ai."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

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
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

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
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_missing():
        raise FileNotFoundError("~/.netrc")

    monkeypatch.setattr(netrc, "netrc", _raise_missing)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

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
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

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
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]  # the sha at the end of git+https://…@<sha>
    spec = cloud_lerobot_spec("act")
    assert ref in spec
    assert "latest" not in spec


def test_cloud_lerobot_spec_uses_archive_tarball_not_git() -> None:
    """A GitHub git+ pin is rewritten to the source archive tarball so pip in
    the container can install it without a git binary."""
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    spec = cloud_lerobot_spec("act")
    assert "git+" not in spec
    # 0.6.0 pins by tag (v0.6.0), not a hex SHA, so the archive ref may contain
    # non-hex chars (v, dots). Match any non-slash ref, not just [0-9a-f].
    assert re.search(r"@ https://github\.com/.+/archive/[^/]+\.tar\.gz$", spec)


def test_cloud_lerobot_spec_drops_host_only_extras_and_adds_policy_extra() -> None:
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

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
    """Running from a source tree without installed MakerMods Lab metadata must still
    derive the pin — from pyproject.toml directly."""
    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "requires", lambda name: None)
    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]
    assert ref in hf_cloud.cloud_lerobot_spec("act")


# -- cloud-boundary config localization (device-leak fix) --


def _request(**overrides):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", **overrides)


def test_localize_forces_flavor_device_over_host_detection() -> None:
    """The host's auto-detected device (mps on a Mac) must never reach a cloud
    job: GPU flavors force cuda, cpu tiers force cpu."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud
    from makermodslab.train import build_training_command

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
    from makermodslab.runners.hf_cloud import localize_config_for_cloud
    from makermodslab.train import build_training_command

    config = _request(dataset_root="/Users/someone/.cache/huggingface/lerobot/user/ds")
    localize_config_for_cloud(config, "t4-small")
    assert config.dataset_root is None
    assert "--dataset.root" not in build_training_command(config, "/tmp/out")


def test_localize_rejects_resume_from_host_checkpoint() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

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
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_device == "cuda"


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    local = _request(policy_pretrained_path="/host/checkpoints/000500/pretrained_model")
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        localize_config_for_cloud(local, "t4-small")

    hub = _request(policy_pretrained_path="user/some-model")
    localize_config_for_cloud(hub, "t4-small")  # no raise
    assert hub.policy_pretrained_path == "user/some-model"


def test_localize_allows_a_step_suffixed_hub_pretrained_ref() -> None:
    """MT2, cloud half: fine-tuning from a SPECIFIC Hub step travels as the ref
    'repo@checkpoints/<step_dir>' — the wrapper materializes it pod-side, the
    same way cloud resume already downloads its parent checkpoint. It must reach
    the container verbatim, not be rejected as a host path and not be rewritten
    here (a host path would be meaningless on the pod)."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(policy_pretrained_path="user/some-model@checkpoints/003000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_pretrained_path == "user/some-model@checkpoints/003000"


# -- in-container installer ladder (the image's uv venv ships no pip) --


def test_install_plan_prefers_uv() -> None:
    """The lerobot-gpu image's venv is uv-created (no pip module), so uv on
    PATH must win, with --python pinning the install into this interpreter."""
    from makermodslab.runners.hf_cloud import _install_plan

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
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, True, True)
    assert label == "pip"
    assert cmds == [["/py", "-m", "pip", "install", "--no-cache-dir", "spec"]]


def test_install_plan_bootstraps_pip_via_ensurepip_as_last_resort() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, False, True)
    assert label == "ensurepip+pip"
    assert cmds == [
        ["/py", "-m", "ensurepip", "--upgrade"],
        ["/py", "-m", "pip", "install", "--no-cache-dir", "spec"],
    ]


def test_install_plan_reports_no_installer() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    assert _install_plan("spec", "/py", None, False, False) == (None, [])


# -- checkpoint-completeness check (partial-upload race fix) --


def test_checkpoint_step_ready_false_when_step_dir_empty(tmp_path: Path) -> None:
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_false_when_only_config_json_exists(tmp_path: Path) -> None:
    """The exact race this fixes: lerobot's save_checkpoint writes
    pretrained_model/config.json before model.safetensors, and only creates
    training_state/ after all of pretrained_model/ is written. A poll that
    lands in this window must not consider the step ready."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    pretrained_dir = step_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "config.json").write_text("{}")
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_false_when_only_training_state_exists(tmp_path: Path) -> None:
    """Belt-and-suspenders: training_state/ alone (no pretrained_model/config.json,
    e.g. a directory mid-construction from something other than save_checkpoint)
    is not enough either."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    (step_dir / "training_state").mkdir(parents=True)
    assert _checkpoint_step_ready(step_dir) is False


def _write_pretrained(step_dir: Path, scheduler: dict | None = None) -> Path:
    """Populate a fully-written pretrained_model/ for `step_dir` (config.json,
    model.safetensors, train_config.json with the given `scheduler` value —
    None mirrors a policy with no scheduler preset, e.g. act; a dict mirrors
    one that has one, e.g. diffusion/pi0/smolvla/vqbet)."""
    pretrained_dir = step_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "config.json").write_text("{}")
    (pretrained_dir / "model.safetensors").write_bytes(b"fake-weights")
    (pretrained_dir / "train_config.json").write_text(json.dumps({"scheduler": scheduler}))
    return pretrained_dir


def test_checkpoint_step_ready_false_when_training_state_freshly_created(tmp_path: Path) -> None:
    """The race the original fix left open: lerobot's save_training_state does
    `training_state/.mkdir(...)` *before* writing any of training_step.json /
    rng_state.safetensors / optimizer_state.safetensors / scheduler_state.json.
    A poll landing right after the mkdir (and before any of those files exist)
    must not consider the step ready, even though pretrained_model/ is fully
    written and training_state/ already exists as a directory."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    _write_pretrained(step_dir)
    (step_dir / "training_state").mkdir()  # freshly created, empty
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_false_when_training_state_missing_optimizer(tmp_path: Path) -> None:
    """Same race, later in the window: training_step.json and rng_state.safetensors
    have landed but optimizer_state.safetensors (always written — lerobot's own
    training loop never calls save_checkpoint with optimizer=None) has not."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    _write_pretrained(step_dir)
    training_state_dir = step_dir / "training_state"
    training_state_dir.mkdir()
    (training_state_dir / "training_step.json").write_text("{}")
    (training_state_dir / "rng_state.safetensors").write_bytes(b"fake-rng")
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_true_for_no_scheduler_policy(tmp_path: Path) -> None:
    """A policy with no scheduler preset (train_config.json's "scheduler" is
    null, e.g. act) is ready once training_step.json, rng_state.safetensors,
    and optimizer_state.safetensors exist — save_training_state never writes
    scheduler_state.json for these, so requiring it would leave the step
    permanently un-uploadable."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    _write_pretrained(step_dir, scheduler=None)
    training_state_dir = step_dir / "training_state"
    training_state_dir.mkdir()
    (training_state_dir / "training_step.json").write_text("{}")
    (training_state_dir / "rng_state.safetensors").write_bytes(b"fake-rng")
    (training_state_dir / "optimizer_state.safetensors").write_bytes(b"fake-optim")
    assert _checkpoint_step_ready(step_dir) is True


def test_checkpoint_step_ready_false_for_scheduler_policy_missing_scheduler_state(
    tmp_path: Path,
) -> None:
    """A policy that does have a scheduler preset (e.g. diffusion/pi0/smolvla/
    vqbet) is NOT ready until scheduler_state.json lands too, even though
    optimizer_state.safetensors already exists — the narrower race the
    original fix's is_dir() check would have missed entirely."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    _write_pretrained(step_dir, scheduler={"type": "cosine_decay_with_warmup"})
    training_state_dir = step_dir / "training_state"
    training_state_dir.mkdir()
    (training_state_dir / "training_step.json").write_text("{}")
    (training_state_dir / "rng_state.safetensors").write_bytes(b"fake-rng")
    (training_state_dir / "optimizer_state.safetensors").write_bytes(b"fake-optim")
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_true_for_scheduler_policy_once_complete(tmp_path: Path) -> None:
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    _write_pretrained(step_dir, scheduler={"type": "cosine_decay_with_warmup"})
    training_state_dir = step_dir / "training_state"
    training_state_dir.mkdir()
    (training_state_dir / "training_step.json").write_text("{}")
    (training_state_dir / "rng_state.safetensors").write_bytes(b"fake-rng")
    (training_state_dir / "optimizer_state.safetensors").write_bytes(b"fake-optim")
    (training_state_dir / "scheduler_state.json").write_text("{}")
    assert _checkpoint_step_ready(step_dir) is True


def test_checkpoint_step_ready_fails_closed_when_train_config_unreadable(tmp_path: Path) -> None:
    """If train_config.json is missing or unparsable, don't guess: require
    scheduler_state.json too (fail closed) rather than risk uploading a step
    whose training_state/ is still incomplete for a scheduler-having policy."""
    from makerlab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    pretrained_dir = step_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "config.json").write_text("{}")
    (pretrained_dir / "model.safetensors").write_bytes(b"fake-weights")
    # no train_config.json written at all
    training_state_dir = step_dir / "training_state"
    training_state_dir.mkdir()
    (training_state_dir / "training_step.json").write_text("{}")
    (training_state_dir / "rng_state.safetensors").write_bytes(b"fake-rng")
    (training_state_dir / "optimizer_state.safetensors").write_bytes(b"fake-optim")
    assert _checkpoint_step_ready(step_dir) is False


def test_wrapper_source_inlines_the_tested_checkpoint_ready_check() -> None:
    """The wrapper's checkpoint-completeness check is _checkpoint_step_ready's
    source inlined verbatim, so the in-container upload gate is exactly the
    function the tests above exercise — and _scan_and_upload must call it
    instead of checking config.json directly (that was the bug: uploading and
    marking a step `seen` the moment config.json appeared, even though
    model.safetensors and training_state/ might not exist yet)."""
    import inspect

    from makerlab.runners.hf_cloud import WRAPPER_SOURCE, _checkpoint_step_ready

    assert inspect.getsource(_checkpoint_step_ready) in WRAPPER_SOURCE
    assert "__CHECKPOINT_READY_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert "_checkpoint_step_ready(entry)" in WRAPPER_SOURCE


# -- wrapper sanity --


def test_wrapper_source_compiles_and_launches_an_argv_list() -> None:
    """The wrapper must pass the trainer argv to Popen as a LIST (splitting a
    joined string was the bug-3 hypothesis — it is not the case and must stay
    that way) and quote its log line so spaced values read unambiguously."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # syntactically valid
    assert "subprocess.Popen(list(trainer_argv)" in WRAPPER_SOURCE
    assert "shlex.join(trainer_argv)" in WRAPPER_SOURCE
    assert re.search(r"^import .*\bshlex\b", WRAPPER_SOURCE, re.MULTILINE)  # imported up top


def test_wrapper_source_handles_resume_download() -> None:
    """Cloud resume: the wrapper must parse --resume-from, download the parent
    checkpoint tree, refuse when training_state/ is absent, and pre-seed `seen`
    so it never re-uploads the checkpoint it just pulled down."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # still valid with the resume block
    assert "--resume-from=" in WRAPPER_SOURCE
    assert "snapshot_download" in WRAPPER_SOURCE
    assert "training_state" in WRAPPER_SOURCE
    assert "seen.add(step_dir)" in WRAPPER_SOURCE


def test_wrapper_source_materializes_a_step_suffixed_finetune_base() -> None:
    """MT2, container half: a --policy.pretrained_path naming a Hub STEP is
    downloaded pod-side and rewritten to that local dir before the trainer runs.

    Two properties the block must keep:
      * it pulls ONLY that step's pretrained_model/ (weights-only — fine-tuning
        needs no training_state/), and
      * it uses the snapshot cache rather than <output_dir>/checkpoints/, which
        the uploader watches — a base checkpoint copied there would be
        republished as if this run had produced it.

    Source-level assertions: the block is top-level wrapper code (like the
    resume download it mirrors), so it has no import seam to exec against. The
    argv rewrite it depends on IS unit-tested below."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")
    assert '_arg("--policy.pretrained_path")' in WRAPPER_SOURCE
    assert '_set_arg("--policy.pretrained_path", str(base_dir))' in WRAPPER_SOURCE
    assert 'allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"]' in WRAPPER_SOURCE
    # Never staged under the watched output dir.
    assert 'base_dir = Path(local_root) / "checkpoints" / step_dir / "pretrained_model"' in WRAPPER_SOURCE


def _wrapper_argv_helpers(trainer_argv: list[str]):
    """Exec the wrapper's own `_arg` / `_set_arg` over `trainer_argv`.

    Sliced out of WRAPPER_SOURCE by name and given the globals the wrapper would
    have, so a drift between the template and these tests fails loudly instead
    of silently testing a host-side paraphrase."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    namespace: dict = {"trainer_argv": trainer_argv}
    for name in ("_arg", "_set_arg"):
        match = re.search(rf"^def {name}\(.*?(?=^\S)", WRAPPER_SOURCE, re.MULTILINE | re.DOTALL)
        assert match, f"{name} not found in WRAPPER_SOURCE"
        exec(compile(match.group(0), "<hf-jobs-wrapper>", "exec"), namespace)  # noqa: S102
    return namespace["_arg"], namespace["_set_arg"]


def test_wrapper_set_arg_rewrites_both_argv_spellings() -> None:
    """The rewrite must hit whichever form the argv builder used, and touch
    nothing else — this is what turns a Hub ref only the pod can resolve into a
    real path for the trainer."""
    joined = ["--policy.type=act", "--policy.pretrained_path=user/repo@checkpoints/003000", "--steps=10"]
    arg, set_arg = _wrapper_argv_helpers(joined)
    assert arg("--policy.pretrained_path") == "user/repo@checkpoints/003000"
    assert set_arg("--policy.pretrained_path", "/tmp/base") is True
    assert joined == ["--policy.type=act", "--policy.pretrained_path=/tmp/base", "--steps=10"]

    split = ["--policy.pretrained_path", "user/repo@checkpoints/003000", "--steps", "10"]
    arg, set_arg = _wrapper_argv_helpers(split)
    assert arg("--policy.pretrained_path") == "user/repo@checkpoints/003000"
    assert set_arg("--policy.pretrained_path", "/tmp/base") is True
    assert split == ["--policy.pretrained_path", "/tmp/base", "--steps", "10"]


def test_wrapper_set_arg_reports_a_missing_flag() -> None:
    """A bare-repo-id fine-tune (or no fine-tune at all) leaves the argv alone;
    the wrapper treats an unexpected miss as a hard error rather than launching
    a run that trains from the wrong weights."""
    argv = ["--policy.type=act"]
    arg, set_arg = _wrapper_argv_helpers(argv)
    assert arg("--policy.pretrained_path") is None
    assert set_arg("--policy.pretrained_path", "/tmp/base") is False
    assert argv == ["--policy.type=act"]


def test_cloud_resume_argv_keeps_lineage_in_parent_repo() -> None:
    """A cloud-resume config resolves to a --config_path at the container path and
    pushes into the parent's repo (same lineage), with resume essentials only."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/ds",
        resume=True,
        steps=20000,
        policy_push_to_hub=True,
        policy_repo_id="user/parent-run",
        config_path="/tmp/makermodslab/train/checkpoints/005000/pretrained_model/train_config.json",
    )
    cmd = build_training_command(req, output_dir="/tmp/makermodslab/train")
    assert (
        "--config_path=/tmp/makermodslab/train/checkpoints/005000/pretrained_model/train_config.json" in cmd
    )
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

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE, _install_plan

    assert inspect.getsource(_install_plan) in WRAPPER_SOURCE
    assert "__INSTALL_PLAN_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert 'shutil.which("uv")' in WRAPPER_SOURCE
    assert "no uv, pip, or ensurepip" in WRAPPER_SOURCE  # clear terminal message


# ---------------------------------------------------------------------------
# Job-timeout precedence: request value wins (normalised to seconds), else the
# HF_JOB_TIMEOUT fallback constant.
# ---------------------------------------------------------------------------


def test_resolve_job_timeout_falls_back_to_constant_when_unset() -> None:
    from makermodslab.runners.hf_cloud import HF_JOB_TIMEOUT, resolve_job_timeout
    from makermodslab.train import TrainingRequest

    config = TrainingRequest(dataset_repo_id="x")
    assert config.hf_job_timeout is None
    assert resolve_job_timeout(config) == HF_JOB_TIMEOUT  # "2h" string passthrough


def test_resolve_job_timeout_uses_request_value_normalised_to_seconds() -> None:
    """An explicit request value wins over the constant and is converted to an
    int of seconds — run_job's own str parser only handles a single unit, so
    compound forms like "3h30m" must be pre-resolved here."""
    from makermodslab.runners.hf_cloud import resolve_job_timeout
    from makermodslab.train import TrainingRequest

    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="45m")) == 2700
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="3h30m")) == 12600
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="2h")) == 7200
