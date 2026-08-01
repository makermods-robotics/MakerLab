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
"""Tests for makerlab.runners.hf_cloud — covers the host-side wandb credential
resolution, the pinned-lerobot spec derivation, and the cloud-boundary config
localization, plus the terminal-stage bookkeeping in HfCloudJobRunner.stop()
(which decides whether a stopped run reads as `interrupted` or as a failure,
and needs only a two-method fake). Submission, log tailing and status polling
talk to HF Jobs and are intentionally left for integration tests."""

from __future__ import annotations

import netrc
import re
import tomllib
from pathlib import Path

import pytest


def test_resolve_wandb_api_key_prefers_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makerlab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.setenv("WANDB_API_KEY", "env-key-123")
    assert resolve_wandb_api_key() == "env-key-123"


def test_resolve_wandb_api_key_falls_back_to_netrc(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WANDB_API_KEY is unset, the function must read the same place
    `wandb login` writes — ~/.netrc under machine api.wandb.ai."""
    from makerlab.runners.hf_cloud import resolve_wandb_api_key

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
    from makerlab.runners.hf_cloud import resolve_wandb_api_key

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
    from makerlab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_missing():
        raise FileNotFoundError("~/.netrc")

    monkeypatch.setattr(netrc, "netrc", _raise_missing)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makerlab.runners.hf_cloud import resolve_wandb_api_key

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
    from makerlab.runners.hf_cloud import resolve_wandb_api_key

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
    from makerlab.runners.hf_cloud import cloud_lerobot_spec

    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]  # the sha at the end of git+https://…@<sha>
    spec = cloud_lerobot_spec("act")
    assert ref in spec
    assert "latest" not in spec


def test_cloud_lerobot_spec_uses_archive_tarball_not_git() -> None:
    """A GitHub git+ pin is rewritten to the source archive tarball so pip in
    the container can install it without a git binary."""
    from makerlab.runners.hf_cloud import cloud_lerobot_spec

    spec = cloud_lerobot_spec("act")
    assert "git+" not in spec
    # 0.6.0 pins by tag (v0.6.0), not a hex SHA, so the archive ref may contain
    # non-hex chars (v, dots). Match any non-slash ref, not just [0-9a-f].
    assert re.search(r"@ https://github\.com/.+/archive/[^/]+\.tar\.gz$", spec)


def test_cloud_lerobot_spec_drops_host_only_extras_and_adds_policy_extra() -> None:
    from makerlab.runners.hf_cloud import cloud_lerobot_spec

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
    """Running from a source tree without installed MakerLab metadata must still
    derive the pin — from pyproject.toml directly."""
    from makerlab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "requires", lambda name: None)
    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]
    assert ref in hf_cloud.cloud_lerobot_spec("act")


# -- cloud-boundary config localization (device-leak fix) --


def _request(**overrides):
    from makerlab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", **overrides)


def test_localize_forces_flavor_device_over_host_detection() -> None:
    """The host's auto-detected device (mps on a Mac) must never reach a cloud
    job: GPU flavors force cuda, cpu tiers force cpu."""
    from makerlab.runners.hf_cloud import localize_config_for_cloud
    from makerlab.train import build_training_command

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
    from makerlab.runners.hf_cloud import localize_config_for_cloud
    from makerlab.train import build_training_command

    config = _request(dataset_root="/Users/someone/.cache/huggingface/lerobot/user/ds")
    localize_config_for_cloud(config, "t4-small")
    assert config.dataset_root is None
    assert "--dataset.root" not in build_training_command(config, "/tmp/out")


def test_localize_rejects_resume_from_host_checkpoint() -> None:
    from makerlab.runners.hf_cloud import localize_config_for_cloud

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
    from makerlab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_device == "cuda"


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from makerlab.runners.hf_cloud import localize_config_for_cloud

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
    from makerlab.runners.hf_cloud import _install_plan

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
    from makerlab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, True, True)
    assert label == "pip"
    assert cmds == [["/py", "-m", "pip", "install", "--no-cache-dir", "spec"]]


def test_install_plan_bootstraps_pip_via_ensurepip_as_last_resort() -> None:
    from makerlab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, False, True)
    assert label == "ensurepip+pip"
    assert cmds == [
        ["/py", "-m", "ensurepip", "--upgrade"],
        ["/py", "-m", "pip", "install", "--no-cache-dir", "spec"],
    ]


def test_install_plan_reports_no_installer() -> None:
    from makerlab.runners.hf_cloud import _install_plan

    assert _install_plan("spec", "/py", None, False, False) == (None, [])


# -- wrapper sanity --


def test_wrapper_source_compiles_and_launches_an_argv_list() -> None:
    """The wrapper must pass the trainer argv to Popen as a LIST (splitting a
    joined string was the bug-3 hypothesis — it is not the case and must stay
    that way) and quote its log line so spaced values read unambiguously."""
    from makerlab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # syntactically valid
    assert "subprocess.Popen(list(trainer_argv)" in WRAPPER_SOURCE
    assert "shlex.join(trainer_argv)" in WRAPPER_SOURCE
    assert re.search(r"^import .*\bshlex\b", WRAPPER_SOURCE, re.MULTILINE)  # imported up top


def test_wrapper_source_handles_resume_download() -> None:
    """Cloud resume: the wrapper must parse --resume-from, download the parent
    checkpoint tree, refuse an incomplete one using the same readiness rule the
    uploader applies, and pre-seed `seen` so it never re-uploads the checkpoint
    it just pulled down."""
    from makerlab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # still valid with the resume block
    assert "--resume-from=" in WRAPPER_SOURCE
    assert "snapshot_download" in WRAPPER_SOURCE
    assert "training_state" in WRAPPER_SOURCE
    assert "missing_checkpoint_files(scan_checkpoint_dir(dest)[0])" in WRAPPER_SOURCE
    assert "seen.add(step_dir)" in WRAPPER_SOURCE


# ---------------------------------------------------------------------------
# Checkpoint uploader (NEW-17). The watcher used to publish a directory the
# moment pretrained_model/config.json appeared — the FIRST file of a save that
# takes seconds — and `seen.add` then retired it forever, so the files that
# finished writing afterwards were never uploaded. The tests below exec the
# wrapper's own _scan_and_upload against fakes, so they exercise the exact
# source that runs in the container rather than a host-side paraphrase.
# ---------------------------------------------------------------------------


class _FakeUploadApi:
    """Records upload_folder calls; optionally fails the first `fail_times`."""

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[dict] = []
        self._fail_times = fail_times

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("hub is having a day")

    @property
    def uploaded_steps(self) -> list[str]:
        return [c["path_in_repo"].rsplit("/", 1)[-1] for c in self.calls]


def _wrapper_scanner(output_dir: Path, api: _FakeUploadApi):
    """Exec the wrapper's own `_scan_and_upload` and return (call, seen).

    The function is sliced out of WRAPPER_SOURCE by name and given the globals
    the wrapper would have around it, so a drift between the template and this
    test surfaces as a KeyError/NameError rather than passing silently.
    """
    from makerlab.jobs import missing_checkpoint_files, scan_checkpoint_dir
    from makerlab.runners.hf_cloud import WRAPPER_SOURCE

    match = re.search(r"^def _scan_and_upload\(.*?(?=^\S)", WRAPPER_SOURCE, re.MULTILINE | re.DOTALL)
    assert match, "_scan_and_upload not found in WRAPPER_SOURCE"
    namespace: dict = {
        "Path": Path,
        "re": re,
        "api": api,
        "output_dir": str(output_dir),
        "repo_id": "user/run",
        "seen": set(),
        "pending": {},
        "scan_checkpoint_dir": scan_checkpoint_dir,
        "missing_checkpoint_files": missing_checkpoint_files,
        "print": lambda *a, **k: None,  # keep the wrapper's logging out of pytest
    }
    exec(compile(match.group(0), "<hf-jobs-wrapper>", "exec"), namespace)  # noqa: S102
    return namespace["_scan_and_upload"], namespace["seen"]


def _write_checkpoint(output_dir: Path, step_dir: str, *, with_optimizer: bool = True) -> Path:
    """A lerobot checkpoint tree under <output_dir>/checkpoints/<step_dir>."""
    ck = output_dir / "checkpoints" / step_dir
    pm = ck / "pretrained_model"
    pm.mkdir(parents=True, exist_ok=True)
    (pm / "config.json").write_text("{}")
    (pm / "model.safetensors").write_bytes(b"weights")
    (pm / "train_config.json").write_text("{}")
    ts = ck / "training_state"
    ts.mkdir(exist_ok=True)
    (ts / "training_step.json").write_text("{}")
    (ts / "rng_state.safetensors").write_bytes(b"rng")
    if with_optimizer:
        (ts / "optimizer_state.safetensors").write_bytes(b"optim")
    return ck


def test_wrapper_does_not_upload_or_seal_a_mid_save_checkpoint(tmp_path: Path) -> None:
    """The live failure: a poll tick lands inside the save, before the big
    optimizer file. Nothing may be published, and — the part that made it
    permanent — the step must NOT enter `seen`, so a later tick re-evaluates."""
    api = _FakeUploadApi()
    scan, seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "005000", with_optimizer=False)
    scan()
    scan()  # settled, but still incomplete — completeness is not a timing question
    assert api.calls == []
    assert seen == set()

    # The save finishes; the next two polls (tree changed, then stable) publish it.
    training_state = tmp_path / "checkpoints" / "005000" / "training_state"
    (training_state / "optimizer_state.safetensors").write_bytes(b"optim")
    scan()
    assert api.calls == []  # fingerprint moved — defer one tick
    scan()
    assert api.uploaded_steps == ["005000"]
    assert seen == {"005000"}


def test_wrapper_defers_a_complete_checkpoint_until_the_tree_stops_changing(
    tmp_path: Path,
) -> None:
    """Two-poll stability: the full file set can be present while the last file
    is still being written, so one unchanged poll is required before upload."""
    api = _FakeUploadApi()
    scan, seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "001000")
    scan()
    assert api.calls == []  # first sighting is never enough
    assert seen == set()
    scan()
    assert api.uploaded_steps == ["001000"]

    scan()  # already seen — no duplicate commit
    assert api.uploaded_steps == ["001000"]


def test_wrapper_final_pass_skips_the_settle_wait(tmp_path: Path) -> None:
    """After the trainer exits there is no next poll, so the final pass must
    upload a complete checkpoint on first sight or the last one is lost."""
    api = _FakeUploadApi()
    scan, seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "015000")
    scan(final=True)
    assert api.uploaded_steps == ["015000"]
    assert seen == {"015000"}


def test_wrapper_final_pass_still_refuses_an_incomplete_checkpoint(tmp_path: Path) -> None:
    """Skipping the settle wait must not become skipping the completeness gate:
    a save the trainer died inside stays unpublished."""
    api = _FakeUploadApi()
    scan, seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "004000", with_optimizer=False)
    scan(final=True)
    assert api.calls == []
    assert seen == set()


def test_wrapper_does_not_seal_a_checkpoint_whose_upload_failed(tmp_path: Path) -> None:
    """`seen.add` used to run inside the try, so a step could be retired on a
    partial or failed commit. A failure must leave it retryable."""
    api = _FakeUploadApi(fail_times=1)
    scan, seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "002000")
    scan()
    scan()  # settled ⇒ attempts the upload, which raises
    assert len(api.calls) == 1
    assert seen == set()  # not retired

    scan()  # retried on the next tick, and this time it lands
    assert len(api.calls) == 2
    assert seen == {"002000"}


def test_wrapper_upload_excludes_safetensors_temp_files(tmp_path: Path) -> None:
    """A .tmp* file caught mid-rename has landed on the Hub before; keep it out
    of the commit even when the rest of the tree is complete."""
    api = _FakeUploadApi()
    scan, _seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "003000")
    scan()
    scan()
    assert api.calls[0]["ignore_patterns"] == [".tmp*", "**/.tmp*"]


def test_wrapper_source_inlines_the_tested_checkpoint_readiness_helpers() -> None:
    """The container's readiness rule is jobs.py's source inlined verbatim, the
    same contract the resume guards apply — one definition, three call sites."""
    import inspect

    from makerlab.jobs import missing_checkpoint_files, scan_checkpoint_dir
    from makerlab.runners.hf_cloud import WRAPPER_SOURCE

    assert inspect.getsource(scan_checkpoint_dir) in WRAPPER_SOURCE
    assert inspect.getsource(missing_checkpoint_files) in WRAPPER_SOURCE
    assert "__CHECKPOINT_READINESS_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced


def test_cloud_resume_argv_keeps_lineage_in_parent_repo() -> None:
    """A cloud-resume config resolves to a --config_path at the container path and
    pushes into the parent's repo (same lineage), with resume essentials only."""
    from makerlab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/ds",
        resume=True,
        steps=20000,
        policy_push_to_hub=True,
        policy_repo_id="user/parent-run",
        config_path="/tmp/makerlab/train/checkpoints/005000/pretrained_model/train_config.json",
    )
    cmd = build_training_command(req, output_dir="/tmp/makerlab/train")
    assert "--config_path=/tmp/makerlab/train/checkpoints/005000/pretrained_model/train_config.json" in cmd
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

    from makerlab.runners.hf_cloud import WRAPPER_SOURCE, _install_plan

    assert inspect.getsource(_install_plan) in WRAPPER_SOURCE
    assert "__INSTALL_PLAN_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert 'shutil.which("uv")' in WRAPPER_SOURCE
    assert "no uv, pip, or ensurepip" in WRAPPER_SOURCE  # clear terminal message


# ---------------------------------------------------------------------------
# Job-timeout precedence: request value wins (normalised to seconds), else the
# HF_JOB_TIMEOUT fallback constant.
# ---------------------------------------------------------------------------


def test_resolve_job_timeout_falls_back_to_constant_when_unset() -> None:
    from makerlab.runners.hf_cloud import HF_JOB_TIMEOUT, resolve_job_timeout
    from makerlab.train import TrainingRequest

    config = TrainingRequest(dataset_repo_id="x")
    assert config.hf_job_timeout is None
    assert resolve_job_timeout(config) == HF_JOB_TIMEOUT  # string passthrough, not seconds


def test_hf_job_timeout_constant_is_single_unit_and_covers_measured_runs() -> None:
    """The fallback is handed to run_job as a raw string, and run_job's parser is
    only `float(timeout[:-1]) * factor[timeout[-1]]` — a single unit suffix. A
    compound "improvement" like "1d12h" would not survive that, so pin the shape
    as well as the budget: it must clear the longest run we have measured
    (SmolVLA 15k steps at 2.24 s/step on a10g-small ≈ 8.8h)."""
    from makerlab.runners.hf_cloud import HF_JOB_TIMEOUT

    assert isinstance(HF_JOB_TIMEOUT, str)
    factors = {"s": 1, "m": 60, "h": 3600, "d": 3600 * 24}  # run_job's own table
    assert HF_JOB_TIMEOUT[-1] in factors
    seconds = float(HF_JOB_TIMEOUT[:-1]) * factors[HF_JOB_TIMEOUT[-1]]  # no ValueError
    assert seconds == 24 * 3600
    assert seconds > 8.8 * 3600


def test_resolve_job_timeout_uses_request_value_normalised_to_seconds() -> None:
    """An explicit request value wins over the constant and is converted to an
    int of seconds — run_job's own str parser only handles a single unit, so
    compound forms like "3h30m" must be pre-resolved here."""
    from makerlab.runners.hf_cloud import resolve_job_timeout
    from makerlab.train import TrainingRequest

    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="45m")) == 2700
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="3h30m")) == 12600
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="2h")) == 7200


# --- stop(): distinguishing a cancel from a crash ---------------------------
#
# returncode() collapses every non-COMPLETED stage to 1, so the registry
# classifies cloud runs on terminal_stage() instead. These cover stop()'s own
# decisions only — no submission, no threads, no network — because the stage
# stop() leaves behind is what decides whether a stopped run reads as
# `interrupted` or as a failed model.


class _FakeStatus:
    def __init__(self, stage, message=None):
        self.stage = stage
        self.message = message


class _FakeJobInfo:
    def __init__(self, stage, message=None):
        self.status = _FakeStatus(stage, message)


class _FakeJobsApi:
    """Just the two calls stop() makes."""

    def __init__(self, *, cancel_raises=False, inspect=None, inspect_raises=False):
        self._cancel_raises = cancel_raises
        self._inspect = inspect
        self._inspect_raises = inspect_raises
        self.cancelled = []
        self.inspected = []

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        if self._cancel_raises:
            raise RuntimeError("404 job already ended")

    def inspect_job(self, job_id):
        self.inspected.append(job_id)
        if self._inspect_raises:
            raise RuntimeError("network down")
        return self._inspect


def _runner_with(api, tmp_path, *, stage=None):
    from makerlab.jobs import TrainingMetrics
    from makerlab.runners.hf_cloud import HfCloudJobRunner

    runner = HfCloudJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", "a10g-small")
    runner._api = api
    runner._hf_job_id = "job-1"
    runner._terminal_status = stage
    return runner


def test_stop_records_canceled_stage(tmp_path) -> None:
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert api.cancelled == ["job-1"]
    assert runner.terminal_stage() == "CANCELED"
    assert runner.is_running() is False
    # No corrective lookup needed when the cancel was accepted.
    assert api.inspected == []


def test_stop_does_not_overwrite_a_stage_the_poller_already_saw(tmp_path) -> None:
    """_set_terminal is idempotent, and that is what keeps a run which
    finished before the stop landed reported as COMPLETED."""
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path, stage="COMPLETED")

    runner.stop()

    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0


def test_stop_adopts_the_real_stage_when_cancel_is_refused(tmp_path) -> None:
    """cancel_job refusing usually means the job had ALREADY ended, so the
    pre-set CANCELED describes a run that finished on its own. Re-read it."""
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("COMPLETED"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert api.inspected == ["job-1"]
    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0


def test_stop_adopts_an_error_stage_and_its_message(tmp_path) -> None:
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("ERROR", "Job timeout"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "ERROR"
    assert runner.terminal_message() == "Job timeout"


def test_stop_keeps_canceled_when_the_stage_cannot_be_re_read(tmp_path) -> None:
    """An unreachable Hub leaves CANCELED standing: our cancel is already out,
    so it's the best available account of the run."""
    api = _FakeJobsApi(cancel_raises=True, inspect_raises=True)
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "CANCELED"


def test_stop_keeps_canceled_when_the_job_is_still_running(tmp_path) -> None:
    """cancel_job can also fail transiently. A non-terminal stage is no
    evidence that the run ended on its own, so don't adopt it."""
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("RUNNING"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "CANCELED"


def test_stop_is_a_noop_before_submission(tmp_path) -> None:
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path)
    runner._hf_job_id = None

    runner.stop()

    assert api.cancelled == []
    assert runner.terminal_stage() is None
