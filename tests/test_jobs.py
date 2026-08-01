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
"""Tests for makerlab.jobs — parsers and Pydantic models. Does not exercise
LocalJobRunner.start() (see plan, "Discovered issue")."""

from __future__ import annotations

import json as _json
import os
from pathlib import Path

import pytest


def _make_checkpoint(
    output_dir: Path,
    step: int,
    *,
    with_state: bool = True,
    with_optimizer: bool = True,
) -> None:
    """Lay out a lerobot-style checkpoint under <output_dir>/checkpoints/<step>.

    `with_state=False` is the weights-only shape (an imported model);
    `with_optimizer=False` is the interrupted-save shape the cloud uploader
    used to publish — training_state/ exists but the big optimizer file that
    lerobot writes last never landed.
    """
    ck = output_dir / "checkpoints" / str(step)
    pm = ck / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "config.json").write_text("{}")  # required by _list_local_checkpoints
    (pm / "train_config.json").write_text("{}")
    (pm / "model.safetensors").write_bytes(b"weights")
    if with_state:
        ts = ck / "training_state"
        ts.mkdir()
        (ts / "training_step.json").write_text("{}")
        (ts / "rng_state.safetensors").write_bytes(b"rng")
        if with_optimizer:
            (ts / "optimizer_state.safetensors").write_bytes(b"optim")


def _record(output_dir: Path, runner: str = "local"):
    from makerlab.jobs import JobRecord
    from makerlab.train import TrainingRequest

    return JobRecord(
        id="job-1",
        name="run",
        state="done",
        config=TrainingRequest(dataset_repo_id="user/ds"),
        output_dir=str(output_dir),
        started_at=0.0,
        runner=runner,
    )


def test_resolve_resume_config_path_returns_train_config(tmp_path) -> None:
    from makerlab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 5000)
    path = _resolve_resume_config_path(_record(out), 5000)
    assert path.endswith("checkpoints/5000/pretrained_model/train_config.json")


def test_resolve_resume_config_path_defaults_to_latest(tmp_path) -> None:
    from makerlab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 1000)
    _make_checkpoint(out, 3000)
    path = _resolve_resume_config_path(_record(out), None)  # None ⇒ latest
    assert "checkpoints/3000/" in path


def test_resolve_resume_config_path_rejects_missing_training_state(tmp_path) -> None:
    from makerlab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000, with_state=False)  # weights-only (e.g. imported)
    with pytest.raises(ValueError, match="training_state"):
        _resolve_resume_config_path(_record(out), 2000)


def test_resolve_resume_config_path_rejects_interrupted_save(tmp_path) -> None:
    """training_state/ exists but the optimizer file lerobot writes last never
    landed — the shape the cloud uploader used to publish. It must be refused
    at the API with the remedy named, not accepted and crashed on inside the
    trainer."""
    from makerlab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000, with_optimizer=False)
    with pytest.raises(ValueError, match="incomplete") as excinfo:
        _resolve_resume_config_path(_record(out), 2000)
    assert "optimizer_state.safetensors" in str(excinfo.value)
    assert "fine-tune from its weights" in str(excinfo.value)


def test_resolve_resume_config_path_rejects_non_local(tmp_path) -> None:
    from makerlab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="local"):
        _resolve_resume_config_path(_record(out, runner="hf_cloud"), 2000)


def test_resolve_resume_config_path_rejects_unknown_step(tmp_path) -> None:
    from makerlab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_resume_config_path(_record(out), 9999)


def _cloud_record(repo_id: str | None = "user/act_ds_2026", state: str = "failed"):
    from makerlab.jobs import JobRecord
    from makerlab.train import TrainingRequest

    return JobRecord(
        id="cloud-1",
        name="run",
        state=state,
        config=TrainingRequest(dataset_repo_id="user/ds", steps=10000),
        output_dir="",
        started_at=0.0,
        runner="hf_cloud",
        hf_repo_id=repo_id,
    )


class _FakeHubApi:
    """Minimal HfApi stand-in: returns a fixed repo file listing."""

    def __init__(self, files: list[str]) -> None:
        self._files = files

    def list_repo_files(self, repo_id, repo_type):
        return self._files


def _hub_checkpoint_files(step_dir: str, *, with_optimizer: bool = True) -> list[str]:
    """The repo paths a COMPLETE cloud checkpoint publishes (or, without the
    optimizer file, the partial tree a mid-save upload used to seal)."""
    files = [
        f"checkpoints/{step_dir}/pretrained_model/config.json",
        f"checkpoints/{step_dir}/pretrained_model/model.safetensors",
        f"checkpoints/{step_dir}/pretrained_model/train_config.json",
        f"checkpoints/{step_dir}/training_state/training_step.json",
        f"checkpoints/{step_dir}/training_state/rng_state.safetensors",
    ]
    if with_optimizer:
        files.append(f"checkpoints/{step_dir}/training_state/optimizer_state.safetensors")
    return files


def test_resolve_cloud_resume_returns_repo_and_step_dir(monkeypatch) -> None:
    from makerlab.jobs import _resolve_cloud_resume

    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("005000")))
    repo_id, step_dir = _resolve_cloud_resume(_cloud_record(), 5000)
    assert repo_id == "user/act_ds_2026"
    assert step_dir == "005000"  # zero-padded dir name preserved


def test_resolve_cloud_resume_defaults_to_latest(monkeypatch) -> None:
    from makerlab.jobs import _resolve_cloud_resume

    files = _hub_checkpoint_files("001000") + _hub_checkpoint_files("003000")
    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    _repo, step_dir = _resolve_cloud_resume(_cloud_record(), None)  # None ⇒ latest
    assert step_dir == "003000"


def test_resolve_cloud_resume_rejects_partial_hub_checkpoint(monkeypatch) -> None:
    """The NEW-17 shape: everything on the Hub except the optimizer file the
    uploader raced. `training_state/training_step.json` alone used to pass this
    guard, so the run died inside the trainer on a FileNotFoundError instead of
    at the API with something the user can act on."""
    from makerlab.jobs import _resolve_cloud_resume

    files = _hub_checkpoint_files("005000", with_optimizer=False)
    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="incomplete on the Hub") as excinfo:
        _resolve_cloud_resume(_cloud_record(), 5000)
    message = str(excinfo.value)
    assert "uploader race" in message
    assert "training_state/optimizer_state.safetensors" in message
    assert "fine-tune from its weights" in message  # the named remedy


def test_resolve_cloud_resume_ignores_other_steps_when_checking_completeness(
    monkeypatch,
) -> None:
    """Completeness is judged per step: a complete 001000 must not vouch for a
    partial 003000 (the file listing is repo-wide and flat)."""
    from makerlab.jobs import _resolve_cloud_resume

    files = _hub_checkpoint_files("001000") + _hub_checkpoint_files("003000", with_optimizer=False)
    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="incomplete on the Hub"):
        _resolve_cloud_resume(_cloud_record(), 3000)


def test_resolve_cloud_resume_rejects_no_checkpoints(monkeypatch) -> None:
    from makerlab.jobs import _resolve_cloud_resume

    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(["README.md"]))
    with pytest.raises(ValueError, match="died before its first save"):
        _resolve_cloud_resume(_cloud_record(), None)


def test_resolve_cloud_resume_rejects_missing_training_state(monkeypatch) -> None:
    from makerlab.jobs import _resolve_cloud_resume

    # Weights present but no training_state/ on the Hub ⇒ not resumable.
    files = ["checkpoints/005000/pretrained_model/config.json"]
    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="training_state"):
        _resolve_cloud_resume(_cloud_record(), 5000)


def test_resolve_cloud_resume_rejects_unknown_step(monkeypatch) -> None:
    from makerlab.jobs import _resolve_cloud_resume

    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("005000")))
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_cloud_resume(_cloud_record(), 9999)


def test_resolve_cloud_resume_rejects_non_cloud(tmp_path) -> None:
    from makerlab.jobs import _resolve_cloud_resume

    with pytest.raises(ValueError, match="cloud"):
        _resolve_cloud_resume(_record(tmp_path, runner="local"), None)


def test_resolve_cloud_resume_rejects_missing_repo() -> None:
    from makerlab.jobs import _resolve_cloud_resume

    with pytest.raises(ValueError, match="no output repo"):
        _resolve_cloud_resume(_cloud_record(repo_id=None), None)


# ---------------------------------------------------------------------------
# Checkpoint completeness — the single readiness rule shared by both resume
# guards above and (inlined verbatim) by the in-container cloud uploader.
# ---------------------------------------------------------------------------


def _complete_names() -> set[str]:
    return {
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/train_config.json",
        "training_state/training_step.json",
        "training_state/rng_state.safetensors",
        "training_state/optimizer_state.safetensors",
    }


def test_missing_checkpoint_files_accepts_a_complete_tree() -> None:
    from makerlab.jobs import missing_checkpoint_files

    assert missing_checkpoint_files(_complete_names()) == []


def test_missing_checkpoint_files_does_not_require_a_scheduler() -> None:
    """save_training_state writes scheduler_state.json only `if scheduler is not
    None`, so requiring it would permanently block scheduler-less presets."""
    from makerlab.jobs import missing_checkpoint_files

    assert "training_state/scheduler_state.json" not in _complete_names()
    assert missing_checkpoint_files(_complete_names()) == []


def test_missing_checkpoint_files_flags_a_mid_save_snapshot() -> None:
    """config.json is the FIRST artifact lerobot writes — on its own it means a
    save just started, not a checkpoint."""
    from makerlab.jobs import missing_checkpoint_files

    missing = missing_checkpoint_files({"pretrained_model/config.json"})
    assert "pretrained_model/*.safetensors" in missing
    assert "training_state/training_step.json" in missing
    assert "training_state/optimizer_state.safetensors" in missing


def test_missing_checkpoint_files_flags_the_optimizer_file_alone() -> None:
    from makerlab.jobs import missing_checkpoint_files

    names = _complete_names() - {"training_state/optimizer_state.safetensors"}
    assert missing_checkpoint_files(names) == ["training_state/optimizer_state.safetensors"]


def test_missing_checkpoint_files_accepts_nested_multi_optimizer_state() -> None:
    """A MultiAdam policy writes training_state/<name>/optimizer_state.safetensors,
    so the optimizer probe must match at any depth or such runs would never be
    considered ready."""
    from makerlab.jobs import missing_checkpoint_files

    names = (_complete_names() - {"training_state/optimizer_state.safetensors"}) | {
        "training_state/actor/optimizer_state.safetensors",
        "training_state/critic/optimizer_state.safetensors",
    }
    assert missing_checkpoint_files(names) == []


def test_missing_checkpoint_files_accepts_a_peft_adapter_as_weights() -> None:
    from makerlab.jobs import missing_checkpoint_files

    names = (_complete_names() - {"pretrained_model/model.safetensors"}) | {
        "pretrained_model/adapter_model.safetensors"
    }
    assert missing_checkpoint_files(names) == []


def test_scan_checkpoint_dir_reports_relative_names_and_a_change_sensitive_fingerprint(
    tmp_path,
) -> None:
    from makerlab.jobs import missing_checkpoint_files, scan_checkpoint_dir

    _make_checkpoint(tmp_path, 1000)
    ck = tmp_path / "checkpoints" / "1000"

    names, fingerprint = scan_checkpoint_dir(ck)
    assert "training_state/optimizer_state.safetensors" in names  # posix, relative
    assert missing_checkpoint_files(names) == []
    assert scan_checkpoint_dir(ck)[1] == fingerprint  # stable while nothing writes

    (ck / "training_state" / "optimizer_state.safetensors").write_bytes(b"grown-larger")
    assert scan_checkpoint_dir(ck)[1] != fingerprint  # a byte written moves it


def test_extract_wandb_run_url_finds_canonical_url() -> None:
    from makerlab.jobs import extract_wandb_run_url

    line = "wandb: \U0001f680 View run at https://wandb.ai/me/myproj/runs/abc123 trailing text"
    assert extract_wandb_run_url(line) == "https://wandb.ai/me/myproj/runs/abc123"


def test_extract_wandb_run_url_returns_none_when_absent() -> None:
    from makerlab.jobs import extract_wandb_run_url

    assert extract_wandb_run_url("nothing here") is None
    assert extract_wandb_run_url("https://example.com/runs/abc") is None


def test_parse_duration_handles_mm_ss_and_hh_mm_ss() -> None:
    from makerlab.jobs import _parse_duration

    assert _parse_duration("01:30") == 90
    assert _parse_duration("01:00:00") == 3600
    assert _parse_duration("?") is None
    assert _parse_duration("garbage") is None


def test_parse_metrics_into_extracts_loss_and_step() -> None:
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    line = "INFO ... step:42 smpl:336 loss:0.0123 grdn:1.5 lr:0.0001 ..."
    parse_metrics_into(line, m)

    assert m.current_step == 42
    assert m.current_loss == pytest.approx(0.0123)
    assert m.current_lr == pytest.approx(0.0001)
    assert m.grad_norm == pytest.approx(1.5)


def test_parse_metrics_into_keeps_tqdm_step_when_log_line_step_is_abbreviated() -> None:
    """At >=1000 steps lerobot formats the log-line step with format_big_number
    ("1K"), which int() can't parse. Feeding a tqdm line (exact step) then the
    abbreviated loss line into the same metrics object must retain the exact
    step and still extract the loss — this is what read_metrics_history relies
    on so it doesn't drop every point past step 1000.
    """
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  10%|██░| 1000/10000 [00:30<04:30, 3.2it/s]", m)
    parse_metrics_into("INFO ... step:1K smpl:8K loss:0.0077 grdn:0.9 lr:0.0001 ...", m)

    assert m.current_step == 1000  # kept from tqdm, not zeroed by "1K"
    assert m.current_loss == pytest.approx(0.0077)
    assert m.current_lr == pytest.approx(0.0001)


def _tqdm_burst(first: int, last: int, total: int, eta: str = "6:26:18") -> str:
    """One log line carrying every tqdm redraw from `first` to `last`.

    tqdm separates redraws with \\r; a transport that doesn't split on \\r (HF
    Jobs' SSE log stream) delivers the whole burst as a single line with the
    trailing 'INFO ... step:N ...' appended to the LAST frame.
    """
    return "\r".join(
        f"Training:  39%|███▊      | {s}/{total} [2:31:07<{eta},  2.12s/step]" for s in range(first, last + 1)
    )


@pytest.mark.parametrize(
    ("burst", "info", "resume_total", "expect_step", "expect_total"),
    [
        # The real shape of a resumed cloud run: 50 frames of the remaining-window
        # bar + an abbreviated 'step:4K' that int() can't use. Last frame 50 of
        # 11000 remaining, on a 15000-step target → global step 4050.
        (
            _tqdm_burst(1, 50, 11000),
            "INFO 2026-07-29 02:11:59 train.py:606 step:4K smpl:259K ep:878 "
            "epch:43.90 loss:0.040 grdn:0.919 lr:8.4e-05",
            15000,
            4050,
            15000,
        ),
        # Same batching on a fresh run: the bar is already global, and the
        # 'step:1K' token is still unusable, so the last frame must stand.
        (
            _tqdm_burst(951, 1000, 10000),
            "INFO ... step:1K smpl:8K loss:0.0077 grdn:0.9 lr:0.0001",
            None,
            1000,
            10000,
        ),
        # Below 1000 the log line's step is a plain int and wins outright —
        # which is also the only reason the first-frame bug stayed invisible
        # under step 1000.
        (
            _tqdm_burst(901, 950, 10000),
            "INFO ... step:950 smpl:7K loss:0.0077 grdn:0.9 lr:0.0001",
            None,
            950,
            10000,
        ),
    ],
    ids=["resumed-cloud-burst", "fresh-burst-abbreviated", "fresh-burst-exact"],
)
def test_parse_metrics_into_uses_the_last_tqdm_frame_of_a_batched_line(
    burst: str, info: str, resume_total: int | None, expect_step: int, expect_total: int
) -> None:
    """A batched line's LAST tqdm frame is the one the appended INFO line belongs
    to. Taking the first understated every step above 1000 by log_freq−1 (a real
    run charted 8201 where the true step was 8250)."""
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into(f"{burst}{info}", m, resume_total)

    assert m.current_step == expect_step
    assert m.total_steps == expect_total
    assert m.current_loss is not None
    # ETA comes from the same (last) frame.
    assert m.eta_seconds == 6 * 3600 + 26 * 60 + 18


def test_read_metrics_history_of_a_batched_resumed_log(tmp_path) -> None:
    """End-to-end on the shape a resumed cloud run actually writes: batched tqdm
    bursts + abbreviated step tokens land on the true global steps (multiples of
    log_freq), not log_freq−1 below them."""
    from makerlab.jobs import JobRecord, JobRegistry, LogLine, _job_log_path
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path)
    root = reg._output_root
    msgs = [
        _tqdm_burst(first, first + 49, 11000) + f"INFO ... step:4K loss:0.04{i} grdn:0.9 lr:8.4e-05"
        for i, first in enumerate((1, 51, 101))
    ]
    p = _job_log_path(root, "R")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for msg in msgs:
            f.write(LogLine(timestamp=0.0, message=msg).model_dump_json() + "\n")
    reg._records["R"] = JobRecord(
        id="R",
        name="r",
        state="done",
        config=TrainingRequest(dataset_repo_id="d", resume=True, steps=15000),
        output_dir=str(root / "R" / "run"),
        started_at=0.0,
    )

    assert [pt.step for pt in reg.read_metrics_history("R")] == [4050, 4100, 4150]


def test_parse_metrics_into_extracts_tqdm_progress() -> None:
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    # tqdm format: "Training:  10%|...| 100/1000 [00:30<04:30, ..."
    line = "Training:  10%|██░|  100/1000 [00:30<04:30, 3.21it/s]"
    parse_metrics_into(line, m)

    assert m.current_step == 100
    assert m.total_steps == 1000
    assert m.eta_seconds == 270  # 4 min 30 s


def test_parse_metrics_into_rebases_resumed_tqdm_to_global_step() -> None:
    """On resume lerobot's bar counts only the remaining window (0 → steps−ckpt),
    so a raw 55/100 is really global step 155 of 200. With resume_total set, the
    parser must rebase so the UI shows 155/200, not 55/100."""
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into(
        "Training:  55%|█████| 55/100 [00:30<01:00, 2.0s/step]", m, resume_total=200
    )
    assert m.current_step == 155  # 200 - 100 + 55
    assert m.total_steps == 200


def test_parse_metrics_into_fresh_run_ignores_resume_rebase() -> None:
    """A fresh run passes resume_total=None; its bar is already the global step."""
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  30%|███| 30/100 [00:30<01:00, 2.0s/step]", m)
    assert m.current_step == 30
    assert m.total_steps == 100


def test_read_metrics_history_stitches_resume_lineage(tmp_path) -> None:
    """A resumed run's curve is continuous across the whole lineage: the source
    run's points (0→100) are prepended to the resumed run's (150→200)."""
    from makerlab.jobs import JobRecord, JobRegistry, LogLine, _job_log_path
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path)
    root = reg._output_root

    def write_log(job_id: str, msgs: list[str]) -> None:
        p = _job_log_path(root, job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for m in msgs:
                f.write(LogLine(timestamp=0.0, message=m).model_dump_json() + "\n")

    write_log("A", ["INFO step:50 loss:1.5 grdn:1 lr:0.001", "INFO step:100 loss:1.2 grdn:1 lr:0.001"])
    write_log("B", ["INFO step:150 loss:1.1 grdn:1 lr:5e-4", "INFO step:200 loss:1.0 grdn:1 lr:2e-4"])
    reg._records["A"] = JobRecord(
        id="A", name="a", state="done",
        config=TrainingRequest(dataset_repo_id="d"),
        output_dir=str(root / "A" / "run"), started_at=0.0,
    )
    reg._records["B"] = JobRecord(
        id="B", name="b", state="done",
        config=TrainingRequest(dataset_repo_id="d", resume=True, resume_from_job_id="A", steps=200),
        output_dir=str(root / "B" / "run"), started_at=0.0,
    )

    assert [p.step for p in reg.read_metrics_history("B")] == [50, 100, 150, 200]
    # The source run on its own is unchanged (no lineage to prepend).
    assert [p.step for p in reg.read_metrics_history("A")] == [50, 100]


def test_parse_metrics_into_ignores_unrelated_lines() -> None:
    from makerlab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("just a log line with no metrics", m)
    assert m.current_step == 0 or m.current_step is None  # accept either default


def test_log_line_round_trips_to_json() -> None:
    from makerlab.jobs import LogLine

    line = LogLine(timestamp=1.5, message="hello")
    payload = line.model_dump_json()
    parsed = LogLine.model_validate_json(payload)
    assert parsed.timestamp == 1.5
    assert parsed.message == "hello"


def test_pid_alive_returns_false_for_unlikely_pid() -> None:
    from makerlab.jobs import _pid_alive

    # DISCOVERED: os.kill(-1, 0) on macOS sends to process group and succeeds
    # (returns True), so we use a large PID that certainly does not exist.
    assert _pid_alive(999999999) is False


def test_hub_checkpoints_from_files_parses_tree() -> None:
    from makerlab.jobs import _hub_checkpoints_from_files

    files = [
        "README.md",
        "checkpoints/000010/pretrained_model/config.json",
        "checkpoints/000020/pretrained_model/config.json",
        "checkpoints/000020/pretrained_model/model.safetensors",
    ]
    out = _hub_checkpoints_from_files(files, "user/repo")
    assert [c.step for c in out] == [10, 20]
    assert out[1].source == "hub"
    assert out[1].ref == "user/repo@checkpoints/000020"


def _make_pretrained(dir_path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(_json.dumps({"type": "act"}))


def test_list_imported_local_single_model(tmp_path) -> None:
    from makerlab.jobs import _list_imported_local

    _make_pretrained(tmp_path)  # config.json at the root
    out = _list_imported_local(str(tmp_path))
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "local"
    assert out[0].ref == str(tmp_path.resolve())


def test_list_imported_local_checkpoints_tree(tmp_path) -> None:
    from makerlab.jobs import _list_imported_local

    _make_pretrained(tmp_path / "checkpoints" / "000010" / "pretrained_model")
    out = _list_imported_local(str(tmp_path))
    assert [c.step for c in out] == [10]
    assert out[0].source == "local"
    assert out[0].ref.endswith("/checkpoints/000010/pretrained_model")


def test_list_imported_local_empty_when_no_model(tmp_path) -> None:
    from makerlab.jobs import _list_imported_local

    assert _list_imported_local(str(tmp_path)) == []


def test_list_imported_hub_single_model() -> None:
    from makerlab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors", "README.md"]

    out = _list_imported_hub(FakeApi(), "user/repo")
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "hub"
    assert out[0].ref == "user/repo@root"


def test_list_imported_hub_prefers_checkpoints_tree() -> None:
    from makerlab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return [
                "config.json",  # also present, but the tree wins
                "checkpoints/000050/pretrained_model/config.json",
            ]

    out = _list_imported_hub(FakeApi(), "user/repo")
    assert [c.step for c in out] == [50]
    assert out[0].ref == "user/repo@checkpoints/000050"


def test_list_imported_hub_empty_when_no_model() -> None:
    from makerlab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["README.md"]

    assert _list_imported_hub(FakeApi(), "user/repo") == []


def test_read_checkpoint_config_local_reads_config_json(tmp_path) -> None:
    from makerlab.jobs import JobCheckpoint, _read_checkpoint_config

    (tmp_path / "config.json").write_text(_json.dumps({"type": "act"}))
    ckpt = JobCheckpoint(step=0, source="local", ref=str(tmp_path))
    assert _read_checkpoint_config(ckpt) == {"type": "act"}


def test_read_checkpoint_config_hub_root(monkeypatch, tmp_path) -> None:
    from makerlab.jobs import JobCheckpoint, _read_checkpoint_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"type": "smolvla"}))
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(cfg_file)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ckpt = JobCheckpoint(step=0, source="hub", ref="user/repo@root")
    assert _read_checkpoint_config(ckpt) == {"type": "smolvla"}
    assert seen["repo_id"] == "user/repo"
    assert seen["filename"] == "config.json"


def test_read_checkpoint_config_hub_tree(monkeypatch, tmp_path) -> None:
    from makerlab.jobs import JobCheckpoint, _read_checkpoint_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"type": "act"}))
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(cfg_file)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ckpt = JobCheckpoint(step=50, source="hub", ref="user/repo@checkpoints/000050")
    assert _read_checkpoint_config(ckpt) == {"type": "act"}
    assert seen["repo_id"] == "user/repo"
    assert seen["filename"] == "checkpoints/000050/pretrained_model/config.json"


def test_register_imported_local_dir(tmp_path) -> None:
    from makerlab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)  # config.json at root
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))

    assert rec.runner == "imported"
    assert rec.state == "done"
    assert rec.output_dir == str(model.resolve())
    assert rec.hf_repo_id is None
    cks = reg.list_checkpoints(rec.id)
    assert [c.step for c in cks] == [0]
    # Persisted as a pointer job.json, reloadable.
    reg2 = JobRegistry(tmp_path / "root")
    assert reg2.get(rec.id).runner == "imported"


def test_register_imported_rejects_unusable_source(tmp_path) -> None:
    from makerlab.jobs import JobRegistry

    empty = tmp_path / "empty"
    empty.mkdir()
    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(ValueError, match="No usable model"):
        reg.register_imported(str(empty))


def test_rename_sets_display_name_and_persists(tmp_path) -> None:
    """Rename is a metadata-only alias: trimmed, persisted to job.json, and the
    immutable identity (id / name / output_dir) is untouched."""
    from makerlab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))
    assert rec.display_name is None

    renamed = reg.rename(rec.id, "  pick-and-place v2  ")
    assert renamed.display_name == "pick-and-place v2"  # trimmed
    assert renamed.id == rec.id
    assert renamed.name == rec.name
    assert renamed.output_dir == str(model.resolve())

    # Round-trips through job.json on a fresh registry.
    reg2 = JobRegistry(tmp_path / "root")
    assert reg2.get(rec.id).display_name == "pick-and-place v2"


def test_rename_rejects_empty_and_path_characters(tmp_path) -> None:
    from makerlab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported(str(model))

    with pytest.raises(ValueError, match="empty"):
        reg.rename(rec.id, "   ")
    with pytest.raises(ValueError, match="Invalid"):
        reg.rename(rec.id, "evil/../name")
    assert reg.get(rec.id).display_name is None  # nothing persisted


def test_rename_unknown_job_raises(tmp_path) -> None:
    from makerlab.jobs import JobNotFoundError, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(JobNotFoundError):
        reg.rename("nope", "anything")


def test_rename_allows_duplicate_aliases(tmp_path) -> None:
    """Aliases are display-only (not file keys like calibration/robot names),
    so uniqueness is deliberately NOT enforced."""
    from makerlab.jobs import JobRecord, JobRegistry
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    for jid in ("A", "B"):
        reg._records[jid] = JobRecord(
            id=jid,
            name=jid,
            state="done",
            config=TrainingRequest(dataset_repo_id="d"),
            output_dir=str(reg._output_root / jid / "run"),
            started_at=0.0,
        )
    reg.rename("A", "same alias")
    reg.rename("B", "same alias")
    assert reg.get("A").display_name == "same alias"
    assert reg.get("B").display_name == "same alias"


def test_job_json_without_display_name_loads_with_none(tmp_path) -> None:
    """Registry files written before the alias field existed load fine, and a
    subsequent rename persists the new field alongside the old ones."""
    from makerlab.jobs import JobRegistry

    root = tmp_path / "root"
    job_dir = root / "old-job"
    job_dir.mkdir(parents=True)
    meta = {
        "id": "old-job",
        "name": "ACT · user/ds",
        "state": "done",
        "config": {"dataset_repo_id": "user/ds", "policy_type": "act"},
        "output_dir": str(job_dir / "run"),
        "started_at": 1.0,
    }
    (job_dir / "job.json").write_text(_json.dumps(meta))

    reg = JobRegistry(root)
    assert reg.get("old-job").display_name is None

    reg.rename("old-job", "legacy run")
    data = _json.loads((job_dir / "job.json").read_text())
    assert data["display_name"] == "legacy run"


def test_register_imported_hub_repo(monkeypatch, tmp_path) -> None:
    from makerlab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    # Patch the symbol where jobs.py binds it (`from .utils.hf_auth import
    # shared_hf_api`) — patching it in its home module has no effect on the
    # already-bound name and the test would hit the network.
    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported("user/some-model")

    assert rec.runner == "imported"
    assert rec.hf_repo_id == "user/some-model"
    assert rec.output_dir == ""
    cks = reg.list_checkpoints(rec.id)
    assert [c.ref for c in cks] == ["user/some-model@root"]


def test_register_imported_local_dir_is_idempotent(tmp_path) -> None:
    """Importing the same local dir twice returns the EXISTING record — same
    id, display alias untouched, no second registry entry."""
    from makerlab.jobs import JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported(str(model))
    reg.rename(first.id, "my import")

    again = reg.register_imported(str(model), name="ignored on duplicate")
    assert again.id == first.id
    assert again.display_name == "my import"
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def test_register_imported_hub_repo_is_idempotent(monkeypatch, tmp_path) -> None:
    from makerlab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("user/some-model")
    again = reg.register_imported("user/some-model")
    assert again.id == first.id
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def test_find_imported_hub_id_compare_is_case_insensitive(monkeypatch, tmp_path) -> None:
    """REVERSAL of the earlier exact-match choice, prompted by a real duplicate
    that slipped through on a case-only difference: HF repo ids are practically
    unique case-insensitively (the Hub redirects across casings), and the
    failure mode of exact matching is silent duplicate cards."""
    from makerlab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json"]

    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("user/some-model")
    assert reg.find_imported("user/some-model") is not None
    assert reg.find_imported("User/Some-Model") is not None
    assert reg.register_imported("USER/SOME-MODEL").id == first.id


def test_register_imported_hub_url_normalizes_to_repo_id(monkeypatch, tmp_path) -> None:
    """A pasted model-page URL is normalized to the bare repo id at the boundary
    — both for storage (so checkpoint listing works) and for dedup."""
    from makerlab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "user/some-model"  # bare id, never the pasted URL
            return ["config.json"]

    monkeypatch.setattr("makerlab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("https://huggingface.co/user/some-model/")
    assert first.hf_repo_id == "user/some-model"
    assert reg.register_imported("user/some-model").id == first.id
    assert reg.register_imported("  https://hf.co/user/some-model ").id == first.id
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def _case_variant_dir(path: Path) -> Path | None:
    """A differently-cased spelling of `path` that still resolves to the same
    directory — only possible on a case-insensitive filesystem (macOS default,
    where the real bug happened). None on case-sensitive filesystems."""
    variant = path.parent / path.name.swapcase()
    try:
        if str(variant) != str(path) and variant.is_dir() and os.path.samefile(variant, path):
            return variant
    except OSError:
        pass
    return None


def test_find_imported_local_matches_case_variant_spelling(tmp_path) -> None:
    """Regression from the real duplicate pair: the same directory imported as
    '/Users/mokuroh54/…/smolvla_real_5k/pretrained_model' and
    '/Users/Mokuroh54/…' (case-insensitive macOS filesystem; Path.resolve()
    preserves the typed case) produced two cards, because identity was an
    exact string compare. Identity is now filesystem identity (samefile)."""
    from makerlab.jobs import JobRegistry

    model = tmp_path / "so101-real" / "smolvla_real_5k" / "pretrained_model"
    _make_pretrained(model)
    variant = _case_variant_dir(model)
    if variant is None:
        pytest.skip("requires a case-insensitive filesystem (the real bug's environment)")

    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported(str(model))
    again = reg.register_imported(str(variant))
    assert again.id == first.id
    assert len([r for r in reg.list(limit=100) if r.runner == "imported"]) == 1


def test_boot_sweep_collapses_real_case_variant_duplicate_pair(tmp_path) -> None:
    """Fixture mirrors the real pair found in the live registry:
      smolvla_imported_2026-06-27_16-19-02  name='smolvla 5k'
        output_dir '…/mokuroh54/…/smolvla_real_5k/pretrained_model'
      smolvla_imported_2026-07-02_14-24-15  name='Imported · pretrained_model'
        output_dir '…/Mokuroh54/…' (same directory, different case)
    The sweep groups local imports by device:inode, so the pair collapses to
    the oldest record and the newer job.json-only dir is removed."""
    from makerlab.jobs import JobNotFoundError, JobRegistry

    model = tmp_path / "so101-real" / "smolvla_real_5k" / "pretrained_model"
    _make_pretrained(model)
    variant = _case_variant_dir(model)
    if variant is None:
        pytest.skip("requires a case-insensitive filesystem (the real bug's environment)")

    root = tmp_path / "root"
    _write_imported_pointer(
        root, "smolvla_imported_2026-06-27_16-19-02", str(model), started_at=1782548342.584353
    )
    _write_imported_pointer(
        root, "smolvla_imported_2026-07-02_14-24-15", str(variant), started_at=1782973455.742018
    )

    reg = JobRegistry(root)
    kept = reg.get("smolvla_imported_2026-06-27_16-19-02")
    assert kept.output_dir == str(model)
    with pytest.raises(JobNotFoundError):
        reg.get("smolvla_imported_2026-07-02_14-24-15")
    assert not (root / "smolvla_imported_2026-07-02_14-24-15").exists()


def test_unique_job_id_suffixes_on_same_second_collision(tmp_path, monkeypatch) -> None:
    """_generate_job_id has second-granularity timestamps; two different models
    imported within the same second must not overwrite each other."""
    from makerlab import jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "_generate_job_id", lambda p, d: "act_imported_T")
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_pretrained(a)
    _make_pretrained(b)
    reg = jobs_mod.JobRegistry(tmp_path / "root")
    r1 = reg.register_imported(str(a))
    r2 = reg.register_imported(str(b))
    assert r1.id == "act_imported_T"
    assert r2.id == "act_imported_T-2"
    assert {r.id for r in reg.list(limit=100)} == {r1.id, r2.id}


def _write_imported_pointer(
    root: Path, job_id: str, output_dir: str, started_at: float, display_name: str | None = None
) -> Path:
    """Lay out an on-disk imported pseudo-job dir (job.json only), the way
    older makerlab versions left duplicates behind before dedup-at-registration."""
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    meta = {
        "id": job_id,
        "name": f"Imported · {job_id}",
        "display_name": display_name,
        "state": "done",
        "config": {"dataset_repo_id": "(imported)", "policy_type": "act"},
        "output_dir": output_dir,
        "started_at": started_at,
        "ended_at": started_at,
        "runner": "imported",
    }
    (job_dir / "job.json").write_text(_json.dumps(meta))
    return job_dir


def test_boot_sweep_collapses_duplicate_imports_keeping_oldest(tmp_path) -> None:
    """Pre-existing duplicate pointers collapse on load: oldest kept, the
    newest duplicate's alias migrated onto it, duplicate job.json-only dirs
    removed."""
    from makerlab.jobs import JobNotFoundError, JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    root = tmp_path / "root"
    _write_imported_pointer(root, "A", str(model.resolve()), started_at=1.0)
    _write_imported_pointer(root, "B", str(model.resolve()), started_at=2.0, display_name="nice name")

    reg = JobRegistry(root)
    kept = reg.get("A")
    assert kept.display_name == "nice name"  # migrated from the newer dup
    with pytest.raises(JobNotFoundError):
        reg.get("B")
    assert not (root / "B").exists()  # contained only job.json → removed
    # The migrated alias is persisted on the keeper.
    assert _json.loads((root / "A" / "job.json").read_text())["display_name"] == "nice name"
    # Idempotent: a fresh load sees one record and nothing left to collapse.
    reg2 = JobRegistry(root)
    assert reg2.get("A").display_name == "nice name"


def test_boot_sweep_never_deletes_dirs_with_extra_content(tmp_path) -> None:
    """A duplicate whose dir holds more than job.json is only dropped from the
    in-memory map — its files stay on disk."""
    from makerlab.jobs import JobNotFoundError, JobRegistry

    model = tmp_path / "model"
    _make_pretrained(model)
    root = tmp_path / "root"
    _write_imported_pointer(root, "A", str(model.resolve()), started_at=1.0, display_name="keeper alias")
    dup_dir = _write_imported_pointer(
        root, "B", str(model.resolve()), started_at=2.0, display_name="dup alias"
    )
    (dup_dir / "extra.safetensors").write_text("")  # anything beyond job.json

    reg = JobRegistry(root)
    kept = reg.get("A")
    assert kept.display_name == "keeper alias"  # keeper's own alias wins
    with pytest.raises(JobNotFoundError):
        reg.get("B")
    assert (dup_dir / "job.json").exists()  # nothing deleted
    assert (dup_dir / "extra.safetensors").exists()


def test_flat_feature_dim_reads_single_arm_and_bimanual_state() -> None:
    """observation.state / action are 1-D: [6] for one SO-101 arm, [12] for a
    bimanual (two-arm) checkpoint. The inference modal keys the single-arm vs
    bimanual mismatch off this."""
    from makerlab.jobs import _flat_feature_dim

    assert _flat_feature_dim({"type": "STATE", "shape": [6]}) == 6
    assert _flat_feature_dim({"type": "STATE", "shape": [12]}) == 12
    assert _flat_feature_dim({"type": "ACTION", "shape": (12,)}) == 12


def test_flat_feature_dim_returns_none_for_missing_or_non_1d() -> None:
    from makerlab.jobs import _flat_feature_dim

    assert _flat_feature_dim(None) is None
    assert _flat_feature_dim({}) is None
    assert _flat_feature_dim({"shape": [3, 480, 640]}) is None  # a VISUAL feature
    assert _flat_feature_dim({"shape": []}) is None
    assert _flat_feature_dim({"shape": "nope"}) is None


def test_cloud_start_rejects_local_only_dataset(tmp_path) -> None:
    """A cloud (hf_cloud) run on a dataset that's only local raises
    DatasetNotOnHubError before any record/runner is created — HF Jobs pods
    resolve the dataset from the Hub, so a local-only one would fail remotely."""
    from unittest.mock import patch

    from makerlab.jobs import DatasetNotOnHubError, JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/local_only", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    with (
        patch(
            "makerlab.datasets.get_hub_status",
            return_value={"repo_id": "user/local_only", "status": "local_only", "url": None},
        ),
        pytest.raises(DatasetNotOnHubError) as exc,
    ):
        reg.start(cfg, target)

    assert exc.value.repo_id == "user/local_only"
    assert "not on the Hugging Face Hub" in str(exc.value)
    # Nothing was registered — the guard fires before the record is created.
    assert reg.list(limit=10) == []


def test_cloud_start_allows_hub_dataset(tmp_path) -> None:
    """When the dataset is on the Hub, the preflight passes and the runner is
    started (stubbed here — we assert the guard doesn't block, not a real
    submission)."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/on_hub", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = "https://hf.co/jobs/job-xyz"

    def _fake_runner_factory(*_args, **_kwargs):
        return fake_runner

    with (
        patch(
            "makerlab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makerlab.runners.hf_cloud.HfCloudJobRunner", _fake_runner_factory),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"
    fake_runner.start.assert_called_once()


def test_cloud_start_allows_unknown_status_dataset(tmp_path) -> None:
    """An "unknown" hub status (offline / transient transport error) does NOT
    block the run — a network blip must not wrongly refuse a real Hub dataset;
    the existing _ensure_dataset_on_hub fallback handles a genuinely-missing
    one. The guard only rejects a definitive "local_only"."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/maybe", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    with (
        patch(
            "makerlab.datasets.get_hub_status",
            return_value={"repo_id": "user/maybe", "status": "unknown", "url": None},
        ),
        patch("makerlab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"


def test_cloud_start_passes_resume_total_to_the_runner(tmp_path) -> None:
    """A resumed cloud run must hand the runner its full step target, or the log
    parser can't rebase the remaining-window tqdm bar and the UI reports
    resume-relative progress (observed: 4,251/11,000 instead of 8,251/15,000)."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(
        dataset_repo_id="user/on_hub",
        policy_type="act",
        resume=True,
        # Stands in for a resume selection; the runner (which is what turns this
        # into a Hub download for a cloud job) is stubbed out below.
        config_path="/somewhere/checkpoints/004000/pretrained_model/train_config.json",
        steps=15000,
    )
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    seen: list[tuple] = []
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None
    fake_runner.wandb_run_url.return_value = None  # keep the watchdog's persist clean

    def _factory(*args, **kwargs):
        seen.append(args)
        return fake_runner

    with (
        patch(
            "makerlab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makerlab.runners.hf_cloud.HfCloudJobRunner", _factory),
    ):
        reg.start(cfg, target)

    assert seen and seen[0][-1] == 15000


def test_cloud_reattach_passes_resume_total_to_the_runner(monkeypatch, tmp_path) -> None:
    """Re-attaching to a running cloud job after a restart must carry the resume
    target too — otherwise the progress readout silently rebases itself on the
    remaining window mid-run."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry

    root = tmp_path / "root"
    job_dir = root / "cloud-job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        _json.dumps(
            {
                "id": "cloud-job",
                "name": "SMOLVLA · user/ds",
                "state": "running",
                "config": {
                    "dataset_repo_id": "user/ds",
                    "policy_type": "smolvla",
                    "resume": True,
                    "steps": 15000,
                },
                "output_dir": str(job_dir / "run"),
                "started_at": 1.0,
                "runner": "hf_cloud",
                "hf_job_id": "hf-job-1",
                "hf_flavor": "a10g-small",
            }
        )
    )

    seen: list[tuple] = []

    def _factory(*args, **kwargs):
        seen.append(args)
        return MagicMock()

    # No watchdog: this test is about what _load_from_disk hands the runner, and
    # the tick would poll the (stubbed) runner and the Hub for checkpoints.
    monkeypatch.setattr(JobRegistry, "_start_watchdog", lambda self: None)
    with patch("makerlab.runners.hf_cloud.HfCloudJobRunner", _factory):
        JobRegistry(root)

    assert seen and seen[0][-1] == 15000


def test_local_start_skips_hub_preflight(tmp_path) -> None:
    """A local run on a local-only dataset is fine — no Hub involved — so the
    preflight must not fire (get_hub_status is never consulted)."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/local_only", policy_type="act")

    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242

    with (
        patch("makerlab.datasets.get_hub_status") as get_status,
        patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, JobTarget(runner="local"))

    get_status.assert_not_called()
    assert record.runner == "local"


# ---------------------------------------------------------------------------
# Fine-tune policy-type guard: --policy.type must match the source checkpoint's
# architecture, because lerobot loads pretrained weights non-strictly and would
# otherwise train a fresh policy that only looks like a fine-tune.
# ---------------------------------------------------------------------------


def _finetune_source(policy_type: str, runner: str = "imported"):
    from makerlab.jobs import JobRecord
    from makerlab.train import TrainingRequest

    return JobRecord(
        id="src-1",
        name="Imported · lerobot/smolvla_base",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type=policy_type),
        output_dir="",
        started_at=0.0,
        runner=runner,
    )


def test_check_finetune_policy_type_rejects_mismatch() -> None:
    from makerlab.jobs import _check_finetune_policy_type

    with pytest.raises(ValueError, match="smolvla") as exc:
        _check_finetune_policy_type(_finetune_source("smolvla"), "act")
    # Both sides named, so the toast tells the user what to switch.
    assert "'act'" in str(exc.value)


def test_check_finetune_policy_type_accepts_match() -> None:
    from makerlab.jobs import _check_finetune_policy_type

    _check_finetune_policy_type(_finetune_source("smolvla"), "smolvla")


def test_check_finetune_policy_type_ignores_unknown_source_type() -> None:
    """register_imported records the "model" placeholder when a checkpoint's
    config.json can't be read — that says nothing about the weights, so it must
    not block a fine-tune."""
    from makerlab.jobs import _check_finetune_policy_type

    _check_finetune_policy_type(_finetune_source("model"), "act")


def test_finetune_start_rejects_contradicting_policy_type(tmp_path) -> None:
    """End to end through JobRegistry.start: a smolvla base + an "act" request
    (the old silent default) fails with a 400-shaped ValueError instead of
    launching an ACT run from smolvla weights. No record is created."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    # A flat imported checkpoint dir whose config.json names the architecture.
    src = tmp_path / "smolvla_ckpt"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "smolvla"}))

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))
    assert source.config.policy_type == "smolvla"

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",  # what the form sends when the type never propagates
        finetune_from_job_id=source.id,
    )
    with (
        patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="smolvla"),
    ):
        reg.start(cfg, JobTarget(runner="local"))

    assert [r.id for r in reg.list(limit=10)] == [source.id]


def test_finetune_start_accepts_matching_policy_type(tmp_path) -> None:
    """The same fine-tune with the propagated type launches, and resolves the
    source checkpoint into --policy.pretrained_path."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    src = tmp_path / "smolvla_ckpt"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "smolvla"}))

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="smolvla",
        finetune_from_job_id=source.id,
    )
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(cfg, JobTarget(runner="local"))

    assert record.config.policy_type == "smolvla"
    assert record.config.policy_pretrained_path == str(src.resolve())


# ---------------------------------------------------------------------------
# Checkpoint-level policy-type guard. _check_finetune_policy_type compares the
# source JobRecord's *recorded* type, so it is blind to a request that supplies
# policy_pretrained_path directly (a public TrainingRequest field, no
# finetune_from_job_id needed) and it opts out entirely when the record carries
# register_imported's "model" placeholder. These cover the checkpoint's own
# config.json being consulted instead.
# ---------------------------------------------------------------------------


def _flat_ckpt(tmp_path: Path, name: str, policy_type: str) -> Path:
    """A flat pretrained_model-shaped dir whose config.json names an architecture."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(_json.dumps({"type": policy_type}))
    return d


def test_read_pretrained_policy_type_reads_local_config(tmp_path) -> None:
    from makerlab.jobs import read_pretrained_policy_type

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    assert read_pretrained_policy_type(str(ckpt)) == "smolvla"


def test_read_pretrained_policy_type_none_when_unreadable(tmp_path) -> None:
    """Missing config.json, blank type, and a bad Hub ref all yield None —
    "not established", which callers must not treat as a clean result."""
    from unittest.mock import patch

    from makerlab.jobs import read_pretrained_policy_type

    bare = tmp_path / "no_config"
    bare.mkdir()
    assert read_pretrained_policy_type(str(bare)) is None

    blank = tmp_path / "blank"
    blank.mkdir()
    (blank / "config.json").write_text(_json.dumps({"type": "   "}))
    assert read_pretrained_policy_type(str(blank)) is None

    # Not a directory ⇒ treated as a Hub repo id; a failed download is silent.
    with patch("makerlab.jobs.hf_hub_download", side_effect=OSError("offline")):
        assert read_pretrained_policy_type("someone/nope") is None


def test_check_pretrained_policy_type_rejects_mismatch(tmp_path) -> None:
    from makerlab.jobs import _check_pretrained_policy_type

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    with pytest.raises(ValueError, match="smolvla") as exc:
        _check_pretrained_policy_type(str(ckpt), "act")
    assert "'act'" in str(exc.value)


def test_check_pretrained_policy_type_silent_when_matching_or_unknown(tmp_path) -> None:
    """A match passes, and so does an unverifiable checkpoint — an unreadable
    source must not block a launch, only an actual contradiction may."""
    from makerlab.jobs import _check_pretrained_policy_type

    ckpt = _flat_ckpt(tmp_path, "act_ckpt", "act")
    _check_pretrained_policy_type(str(ckpt), "act")

    bare = tmp_path / "unknown"
    bare.mkdir()
    _check_pretrained_policy_type(str(bare), "act")


def test_start_rejects_direct_pretrained_path_mismatch(tmp_path) -> None:
    """The hole _check_finetune_policy_type leaves open: policy_pretrained_path
    set directly, with no finetune_from_job_id, so the record-based guard never
    runs. The checkpoint's own config.json must still stop it, and no record may
    be created."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    reg = JobRegistry(tmp_path / "root")

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        policy_pretrained_path=str(ckpt),
    )
    with (
        patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="smolvla"),
    ):
        reg.start(cfg, JobTarget(runner="local"))

    assert reg.list(limit=10) == []


def test_start_rejects_finetune_when_record_type_is_placeholder(tmp_path) -> None:
    """register_imported stores the "model" placeholder when it can't read a
    checkpoint's config.json, and _check_finetune_policy_type deliberately skips
    that case. If the config becomes readable by launch time, the checkpoint
    check must still catch the mismatch."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    src = tmp_path / "mystery_ckpt"
    src.mkdir()
    (src / "config.json").write_text(_json.dumps({"type": "smolvla"}))

    reg = JobRegistry(tmp_path / "root")
    source = reg.register_imported(str(src))
    # Simulate the import having failed to read the architecture.
    source.config.policy_type = "model"

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        finetune_from_job_id=source.id,
    )
    with (
        patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="smolvla"),
    ):
        reg.start(cfg, JobTarget(runner="local"))


def test_start_allows_resume_without_checkpoint_type_check(tmp_path) -> None:
    """Resume passes --config_path and never emits --policy.pretrained_path, so
    the pair can't contradict and the guard must not fire on it (a resumed
    smolvla run whose request still carries the "act" default would otherwise be
    refused)."""
    from unittest.mock import MagicMock, patch

    from makerlab.jobs import JobRegistry, JobTarget
    from makerlab.train import TrainingRequest

    ckpt = _flat_ckpt(tmp_path, "smolvla_ckpt", "smolvla")
    reg = JobRegistry(tmp_path / "root")

    cfg = TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        policy_pretrained_path=str(ckpt),
        resume=True,
        config_path=str(tmp_path / "train_config.json"),
    )
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 99
    with patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(cfg, JobTarget(runner="local"))
    assert record.state == "running"


# --- Deliberate stop vs genuine failure -------------------------------------
#
# Regression cover for the defect where every press of Stop landed in run
# history as `failed` + "Subprocess exited with code 1", indistinguishable
# from a crash: JobRegistry.stop() recorded no intent and the watchdog had
# only the exit code to go on. The state machine already had `interrupted`,
# reachable only by startup reconciliation of a stranded record.


class _FakeRunner:
    """Minimal JobRunner. Deliberately does NOT expose the optional hooks —
    subclasses add them, mirroring runners that can and can't answer."""

    def __init__(self, *, code=None, on_stop_code=None, stage=None, on_stop_stage=None):
        self._code = code  # None + no stage => still running
        self._on_stop_code = on_stop_code
        self._stage = stage
        self._on_stop_stage = on_stop_stage
        self.stopped = False

    def start(self, job_id, config, output_dir) -> None:
        # No subprocess: liveness is driven by the fields above so the
        # watchdog's exit-detection can be stepped deterministically.
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self._on_stop_code is not None:
            self._code = self._on_stop_code
        # Idempotent like HfCloudJobRunner._set_terminal: a stage the platform
        # already reported survives our cancel.
        if self._on_stop_stage is not None and self._stage is None:
            self._stage = self._on_stop_stage

    def is_running(self) -> bool:
        return self._code is None and self._stage is None

    def returncode(self):
        if self._stage is not None:
            return 0 if self._stage == "COMPLETED" else 1
        return self._code

    def stream_log_lines(self):
        return []

    def wandb_run_url(self):
        return None

    def pid(self):
        return 4242


class _FakeSignallingRunner(_FakeRunner):
    """A local-shaped runner: reports whether it actually signalled."""

    def __init__(self, *, signals=True, **kw):
        super().__init__(**kw)
        self._signals = signals

    def stop(self) -> None:
        if self._signals:
            super().stop()
        else:
            # Process was already gone; stop() short-circuits and claims
            # nothing, exactly like LocalJobRunner's poll() guard.
            self.stopped = True

    def stop_signalled(self) -> bool:
        return self._signals and self.stopped


class _FakeStagedRunner(_FakeRunner):
    """A cloud-shaped runner: reports a platform terminal stage + message."""

    def __init__(self, *, message=None, **kw):
        super().__init__(**kw)
        self._message = message

    def terminal_stage(self):
        return self._stage

    def terminal_message(self):
        return self._message


def _start_with(reg, runner, **cfg_kw):
    """Start a job whose runner is `runner`, via the real JobRegistry.start."""
    from unittest.mock import patch

    from makerlab.jobs import JobTarget
    from makerlab.train import TrainingRequest

    cfg = TrainingRequest(dataset_repo_id="user/ds", **cfg_kw)
    with patch("makerlab.jobs.LocalJobRunner", lambda *a, **k: runner):
        return reg.start(cfg, JobTarget(runner="local"))


def _stop_and_finalise(reg, job_id):
    """Stop, then force a watchdog tick so the assertion doesn't race the
    1Hz background thread. _tick is a no-op if that thread got there first."""
    reg.stop(job_id)
    reg._tick()
    return reg.get(job_id)


# -- the pure classifier ----------------------------------------------------


@pytest.mark.parametrize(
    ("rc", "stop_requested", "stage", "expected"),
    [
        # Local: a clean exit is `done` no matter what else is true.
        (0, False, None, "done"),
        (0, True, None, "done"),
        # Local: nonzero without a stop is a real failure (unchanged).
        (1, False, None, "failed"),
        (-15, False, None, "failed"),
        # Local: nonzero after a stop we signalled is deliberate.
        (1, True, None, "interrupted"),
        (-15, True, None, "interrupted"),
        # No code at all: no evidence, stays a failure (unchanged).
        (None, False, None, "failed"),
        (None, True, None, "failed"),
        # Cloud: the platform stage wins over the collapsed exit code.
        (0, False, "COMPLETED", "done"),
        (0, True, "COMPLETED", "done"),
        (1, True, "CANCELED", "interrupted"),
        (1, False, "CANCELED", "failed"),
        (1, True, "ERROR", "failed"),
        (1, False, "ERROR", "failed"),
        (1, True, "DELETED", "failed"),
        # Stage matching is case-insensitive (HF returns an enum we str()).
        (1, True, "canceled", "interrupted"),
    ],
)
def test_classify_terminal_state_table(rc, stop_requested, stage, expected) -> None:
    from makerlab.jobs import classify_terminal_state

    assert (
        classify_terminal_state(returncode=rc, stop_requested=stop_requested, terminal_stage=stage)
        == expected
    )


# -- registry: local runner -------------------------------------------------


def test_stop_records_intent_before_signalling(tmp_path) -> None:
    """The intent must be on the registry before the signal leaves, or the
    watchdog can finalise a stop it never heard about."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    seen: list[bool] = []

    class _Probe(_FakeSignallingRunner):
        def stop(self):
            # Observed from inside stop(), i.e. before any signal lands.
            seen.append(record.id in reg._stop_requested)
            super().stop()

    runner = _Probe(on_stop_code=-15)
    record = _start_with(reg, runner)
    reg.stop(record.id)

    assert seen == [True]


def test_local_stop_is_interrupted_not_failed(tmp_path) -> None:
    from makerlab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=-15))

    final = _stop_and_finalise(reg, record.id)
    assert final.state == "interrupted"
    assert final.error_message == STOPPED_BY_REQUEST_MESSAGE
    assert "exited with code" not in (final.error_message or "")
    # The real code is still recorded for anyone debugging.
    assert final.exit_code == -15
    assert final.ended_at is not None


def test_local_stop_of_trainer_that_catches_sigterm_is_still_interrupted(tmp_path) -> None:
    """A trainer with its own SIGTERM handler exits 1, not -15. Narrowing
    `interrupted` to signal-shaped codes would leave the bug unfixed here."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=1))

    assert _stop_and_finalise(reg, record.id).state == "interrupted"


def test_crash_without_a_stop_stays_failed(tmp_path) -> None:
    """The unchanged path: nothing asked this to stop, so it failed."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner()
    record = _start_with(reg, runner)

    runner._code = 1  # crashed on its own
    reg._tick()

    final = reg.get(record.id)
    assert final.state == "failed"
    assert final.error_message == "Subprocess exited with code 1"


def test_clean_finish_racing_a_stop_stays_done(tmp_path) -> None:
    """rc == 0 means the trainer ran its own shutdown to completion; a stop
    that arrived too late must not relabel it."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner(on_stop_code=0)
    record = _start_with(reg, runner)

    final = _stop_and_finalise(reg, record.id)
    assert final.state == "done"
    assert final.error_message is None


def test_crash_before_the_signal_landed_is_not_laundered(tmp_path) -> None:
    """The process died on its own between the intent and the signal, so
    LocalJobRunner.stop() short-circuits and reports it signalled nothing.
    The nonzero code is the process's own: still a failure."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner(signals=False)
    record = _start_with(reg, runner)

    runner._code = 1  # crashed in the window
    final = _stop_and_finalise(reg, record.id)

    assert runner.stopped is True  # we did ask
    assert final.state == "failed"
    assert final.error_message == "Subprocess exited with code 1"


def test_runner_without_the_hook_still_gets_interrupted(tmp_path) -> None:
    """A runner that can't say whether it signalled abstains rather than
    vetoing — recorded intent alone is enough."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeRunner(on_stop_code=1)
    assert not hasattr(runner, "stop_signalled")
    record = _start_with(reg, runner)

    assert _stop_and_finalise(reg, record.id).state == "interrupted"


def test_stop_intent_is_dropped_after_finalisation(tmp_path) -> None:
    """No stale intent may linger to mislabel anything later."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=-15))
    _stop_and_finalise(reg, record.id)

    assert record.id not in reg._stop_requested


def test_interrupted_state_survives_a_restart(tmp_path) -> None:
    """The classification is persisted, not just in-memory — the user's
    history has to still read `interrupted` on the next launch."""
    from makerlab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    root = tmp_path / "root"
    reg = JobRegistry(root)
    record = _start_with(reg, _FakeSignallingRunner(on_stop_code=-15))
    _stop_and_finalise(reg, record.id)
    reg.shutdown()

    reloaded = JobRegistry(root).get(record.id)
    assert reloaded.state == "interrupted"
    assert reloaded.error_message == STOPPED_BY_REQUEST_MESSAGE


def test_stop_rejects_an_already_finished_job_without_recording_intent(tmp_path) -> None:
    from makerlab.jobs import JobNotRunningError, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeSignallingRunner()
    record = _start_with(reg, runner)

    runner._code = 0
    reg._tick()
    assert reg.get(record.id).state == "done"

    with pytest.raises(JobNotRunningError):
        reg.stop(record.id)
    assert record.id not in reg._stop_requested


# -- registry: cloud-shaped runner (classified on terminal_stage) -----------


def test_cloud_cancel_is_interrupted(tmp_path) -> None:
    """The reported case: a stopped HF Jobs run. returncode() collapses every
    non-COMPLETED stage to 1, so before this it read `failed` + "Subprocess
    exited with code 1" and looked like a broken model."""
    from makerlab.jobs import STOPPED_BY_REQUEST_MESSAGE, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    record = _start_with(reg, _FakeStagedRunner(on_stop_stage="CANCELED"))

    final = _stop_and_finalise(reg, record.id)
    assert final.state == "interrupted"
    assert final.error_message == STOPPED_BY_REQUEST_MESSAGE


def test_cloud_job_that_completed_before_the_cancel_stays_done(tmp_path) -> None:
    """The poller saw COMPLETED first; _set_terminal is idempotent so our
    cancel doesn't overwrite it, and the run keeps its success."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner(on_stop_stage="CANCELED")
    record = _start_with(reg, runner)

    runner._stage = "COMPLETED"  # observed by the status poller
    final = _stop_and_finalise(reg, record.id)

    assert final.state == "done"
    assert final.error_message is None


def test_cloud_job_that_errored_before_the_cancel_stays_failed(tmp_path) -> None:
    """A real crash that merely coincided with the stop must not be laundered
    into `interrupted` — that would hide a genuine failure."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner(on_stop_stage="CANCELED", message="boom")
    record = _start_with(reg, runner)

    runner._stage = "ERROR"
    final = _stop_and_finalise(reg, record.id)

    assert final.state == "failed"
    assert final.error_message == "boom"


def test_cloud_timeout_stays_failed_and_keeps_its_platform_message(tmp_path) -> None:
    """HF Jobs' 'Job timeout' arrives as an ERROR stage with a message. It is
    a failure, not a user stop, and the message must still reach the UI."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner(message="Job timeout")
    record = _start_with(reg, runner)

    runner._stage = "ERROR"
    reg._tick()

    final = reg.get(record.id)
    assert final.state == "failed"
    assert final.error_message == "Job timeout"


def test_cloud_cancel_from_outside_makerlab_stays_failed(tmp_path) -> None:
    """A CANCELED we never asked for (HF web UI, platform-side kill). HF's
    stage doesn't say who asked, so this is left alone rather than guessed
    into `interrupted`. Documented limitation, asserted so it's a choice."""
    from makerlab.jobs import JobRegistry

    reg = JobRegistry(tmp_path / "root")
    runner = _FakeStagedRunner()
    record = _start_with(reg, runner)

    runner._stage = "CANCELED"
    reg._tick()

    assert reg.get(record.id).state == "failed"


# -- TailingJobRunner: no Popen to reap, so the code is synthesised ---------


def _tailing_runner(pid, monkeypatch, *, alive=True):
    """A TailingJobRunner over a fake pid; os.kill is stubbed so no real
    process is signalled."""
    from makerlab import jobs as jobs_mod

    state = {"alive": alive}

    def fake_kill(target_pid, sig):
        assert target_pid == pid
        if not state["alive"]:
            raise ProcessLookupError(pid)
        if sig != 0:
            state["alive"] = False  # SIGTERM landed

    monkeypatch.setattr(jobs_mod.os, "kill", fake_kill)
    runner = jobs_mod.TailingJobRunner(jobs_mod.TrainingMetrics(), Path("/nonexistent"), pid)
    return runner, state


def test_tailing_runner_reports_sigterm_after_a_delivered_stop(monkeypatch) -> None:
    """Its returncode() synthesises 0 when the pid is gone, which would file a
    deliberate stop as `done`. Once we know we signalled a live pid, naming
    the signal is the more honest synthetic answer."""
    import signal as signal_mod

    runner, _ = _tailing_runner(31337, monkeypatch)
    assert runner.returncode() is None  # still alive

    runner.stop()
    assert runner.stop_signalled() is True
    assert runner.returncode() == -signal_mod.SIGTERM


def test_tailing_runner_keeps_optimistic_zero_when_pid_was_already_gone(monkeypatch) -> None:
    """Nothing was signalled, so the pid's absence isn't ours to claim: the
    detached run that finished normally still finalises as `done`."""
    runner, _ = _tailing_runner(31338, monkeypatch, alive=False)

    runner.stop()
    assert runner.stop_signalled() is False
    assert runner.returncode() == 0
