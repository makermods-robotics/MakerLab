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
"""Tests for makermodslab.jobs — parsers and Pydantic models. Does not exercise
LocalJobRunner.start() (see plan, "Discovered issue")."""

from __future__ import annotations

import json as _json
import os
from pathlib import Path

import pytest


def _make_checkpoint(output_dir: Path, step: int, *, with_state: bool = True) -> None:
    """Lay out a lerobot-style checkpoint under <output_dir>/checkpoints/<step>."""
    ck = output_dir / "checkpoints" / str(step)
    pm = ck / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "config.json").write_text("{}")  # required by _list_local_checkpoints
    (pm / "train_config.json").write_text("{}")
    if with_state:
        (ck / "training_state").mkdir()


def _record(output_dir: Path, runner: str = "local"):
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

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
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 5000)
    path = _resolve_resume_config_path(_record(out), 5000)
    assert path.endswith("checkpoints/5000/pretrained_model/train_config.json")


def test_resolve_resume_config_path_defaults_to_latest(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 1000)
    _make_checkpoint(out, 3000)
    path = _resolve_resume_config_path(_record(out), None)  # None ⇒ latest
    assert "checkpoints/3000/" in path


def test_resolve_resume_config_path_rejects_missing_training_state(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000, with_state=False)  # weights-only (e.g. imported)
    with pytest.raises(ValueError, match="training_state"):
        _resolve_resume_config_path(_record(out), 2000)


def test_resolve_resume_config_path_rejects_non_local(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="local"):
        _resolve_resume_config_path(_record(out, runner="hf_cloud"), 2000)


def test_resolve_resume_config_path_rejects_unknown_step(tmp_path) -> None:
    from makermodslab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_resume_config_path(_record(out), 9999)


def _cloud_record(repo_id: str | None = "user/act_ds_2026", state: str = "failed"):
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

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


def test_resolve_cloud_resume_returns_repo_and_step_dir(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    files = [
        "checkpoints/005000/pretrained_model/config.json",
        "checkpoints/005000/training_state/training_step.json",
    ]
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    repo_id, step_dir = _resolve_cloud_resume(_cloud_record(), 5000)
    assert repo_id == "user/act_ds_2026"
    assert step_dir == "005000"  # zero-padded dir name preserved


def test_resolve_cloud_resume_defaults_to_latest(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    files = [
        "checkpoints/001000/pretrained_model/config.json",
        "checkpoints/001000/training_state/training_step.json",
        "checkpoints/003000/pretrained_model/config.json",
        "checkpoints/003000/training_state/training_step.json",
    ]
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    _repo, step_dir = _resolve_cloud_resume(_cloud_record(), None)  # None ⇒ latest
    assert step_dir == "003000"


def test_resolve_cloud_resume_rejects_no_checkpoints(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(["README.md"]))
    with pytest.raises(ValueError, match="died before its first save"):
        _resolve_cloud_resume(_cloud_record(), None)


def test_resolve_cloud_resume_rejects_missing_training_state(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    # Weights present but no training_state/ on the Hub ⇒ not resumable.
    files = ["checkpoints/005000/pretrained_model/config.json"]
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="training_state"):
        _resolve_cloud_resume(_cloud_record(), 5000)


def test_resolve_cloud_resume_rejects_unknown_step(monkeypatch) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    files = [
        "checkpoints/005000/pretrained_model/config.json",
        "checkpoints/005000/training_state/training_step.json",
    ]
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_cloud_resume(_cloud_record(), 9999)


def test_resolve_cloud_resume_rejects_non_cloud(tmp_path) -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    with pytest.raises(ValueError, match="cloud"):
        _resolve_cloud_resume(_record(tmp_path, runner="local"), None)


def test_resolve_cloud_resume_rejects_missing_repo() -> None:
    from makermodslab.jobs import _resolve_cloud_resume

    with pytest.raises(ValueError, match="no output repo"):
        _resolve_cloud_resume(_cloud_record(repo_id=None), None)


def test_extract_wandb_run_url_finds_canonical_url() -> None:
    from makermodslab.jobs import extract_wandb_run_url

    line = "wandb: \U0001f680 View run at https://wandb.ai/me/myproj/runs/abc123 trailing text"
    assert extract_wandb_run_url(line) == "https://wandb.ai/me/myproj/runs/abc123"


def test_extract_wandb_run_url_returns_none_when_absent() -> None:
    from makermodslab.jobs import extract_wandb_run_url

    assert extract_wandb_run_url("nothing here") is None
    assert extract_wandb_run_url("https://example.com/runs/abc") is None


def test_parse_duration_handles_mm_ss_and_hh_mm_ss() -> None:
    from makermodslab.jobs import _parse_duration

    assert _parse_duration("01:30") == 90
    assert _parse_duration("01:00:00") == 3600
    assert _parse_duration("?") is None
    assert _parse_duration("garbage") is None


def test_parse_metrics_into_extracts_loss_and_step() -> None:
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

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
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  10%|██░| 1000/10000 [00:30<04:30, 3.2it/s]", m)
    parse_metrics_into("INFO ... step:1K smpl:8K loss:0.0077 grdn:0.9 lr:0.0001 ...", m)

    assert m.current_step == 1000  # kept from tqdm, not zeroed by "1K"
    assert m.current_loss == pytest.approx(0.0077)
    assert m.current_lr == pytest.approx(0.0001)


def test_parse_metrics_into_extracts_tqdm_progress() -> None:
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

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
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into(
        "Training:  55%|█████| 55/100 [00:30<01:00, 2.0s/step]", m, resume_total=200
    )
    assert m.current_step == 155  # 200 - 100 + 55
    assert m.total_steps == 200


def test_parse_metrics_into_fresh_run_ignores_resume_rebase() -> None:
    """A fresh run passes resume_total=None; its bar is already the global step."""
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  30%|███| 30/100 [00:30<01:00, 2.0s/step]", m)
    assert m.current_step == 30
    assert m.total_steps == 100


def test_read_metrics_history_stitches_resume_lineage(tmp_path) -> None:
    """A resumed run's curve is continuous across the whole lineage: the source
    run's points (0→100) are prepended to the resumed run's (150→200)."""
    from makermodslab.jobs import JobRecord, JobRegistry, LogLine, _job_log_path
    from makermodslab.train import TrainingRequest

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
    from makermodslab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("just a log line with no metrics", m)
    assert m.current_step == 0 or m.current_step is None  # accept either default


def test_log_line_round_trips_to_json() -> None:
    from makermodslab.jobs import LogLine

    line = LogLine(timestamp=1.5, message="hello")
    payload = line.model_dump_json()
    parsed = LogLine.model_validate_json(payload)
    assert parsed.timestamp == 1.5
    assert parsed.message == "hello"


def test_pid_alive_returns_false_for_unlikely_pid() -> None:
    from makermodslab.jobs import _pid_alive

    # DISCOVERED: os.kill(-1, 0) on macOS sends to process group and succeeds
    # (returns True), so we use a large PID that certainly does not exist.
    assert _pid_alive(999999999) is False


def test_hub_checkpoints_from_files_parses_tree() -> None:
    from makermodslab.jobs import _hub_checkpoints_from_files

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
    from makermodslab.jobs import _list_imported_local

    _make_pretrained(tmp_path)  # config.json at the root
    out = _list_imported_local(str(tmp_path))
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "local"
    assert out[0].ref == str(tmp_path.resolve())


def test_list_imported_local_checkpoints_tree(tmp_path) -> None:
    from makermodslab.jobs import _list_imported_local

    _make_pretrained(tmp_path / "checkpoints" / "000010" / "pretrained_model")
    out = _list_imported_local(str(tmp_path))
    assert [c.step for c in out] == [10]
    assert out[0].source == "local"
    assert out[0].ref.endswith("/checkpoints/000010/pretrained_model")


def test_list_imported_local_empty_when_no_model(tmp_path) -> None:
    from makermodslab.jobs import _list_imported_local

    assert _list_imported_local(str(tmp_path)) == []


def test_list_imported_hub_single_model() -> None:
    from makermodslab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors", "README.md"]

    out = _list_imported_hub(FakeApi(), "user/repo")
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "hub"
    assert out[0].ref == "user/repo@root"


def test_list_imported_hub_prefers_checkpoints_tree() -> None:
    from makermodslab.jobs import _list_imported_hub

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
    from makermodslab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["README.md"]

    assert _list_imported_hub(FakeApi(), "user/repo") == []


def test_read_checkpoint_config_local_reads_config_json(tmp_path) -> None:
    from makermodslab.jobs import JobCheckpoint, _read_checkpoint_config

    (tmp_path / "config.json").write_text(_json.dumps({"type": "act"}))
    ckpt = JobCheckpoint(step=0, source="local", ref=str(tmp_path))
    assert _read_checkpoint_config(ckpt) == {"type": "act"}


def test_read_checkpoint_config_hub_root(monkeypatch, tmp_path) -> None:
    from makermodslab.jobs import JobCheckpoint, _read_checkpoint_config

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
    from makermodslab.jobs import JobCheckpoint, _read_checkpoint_config

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
    from makermodslab.jobs import JobRegistry

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
    from makermodslab.jobs import JobRegistry

    empty = tmp_path / "empty"
    empty.mkdir()
    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(ValueError, match="No usable model"):
        reg.register_imported(str(empty))


def test_rename_sets_display_name_and_persists(tmp_path) -> None:
    """Rename is a metadata-only alias: trimmed, persisted to job.json, and the
    immutable identity (id / name / output_dir) is untouched."""
    from makermodslab.jobs import JobRegistry

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
    from makermodslab.jobs import JobRegistry

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
    from makermodslab.jobs import JobNotFoundError, JobRegistry

    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(JobNotFoundError):
        reg.rename("nope", "anything")


def test_rename_allows_duplicate_aliases(tmp_path) -> None:
    """Aliases are display-only (not file keys like calibration/robot names),
    so uniqueness is deliberately NOT enforced."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

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
    from makermodslab.jobs import JobRegistry

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
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    # Patch the symbol where jobs.py binds it (`from .utils.hf_auth import
    # shared_hf_api`) — patching it in its home module has no effect on the
    # already-bound name and the test would hit the network.
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
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
    from makermodslab.jobs import JobRegistry

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
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
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
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    first = reg.register_imported("user/some-model")
    assert reg.find_imported("user/some-model") is not None
    assert reg.find_imported("User/Some-Model") is not None
    assert reg.register_imported("USER/SOME-MODEL").id == first.id


def test_register_imported_hub_url_normalizes_to_repo_id(monkeypatch, tmp_path) -> None:
    """A pasted model-page URL is normalized to the bare repo id at the boundary
    — both for storage (so checkpoint listing works) and for dedup."""
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "user/some-model"  # bare id, never the pasted URL
            return ["config.json"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
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
    from makermodslab.jobs import JobRegistry

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
    from makermodslab.jobs import JobNotFoundError, JobRegistry

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
    from makermodslab import jobs as jobs_mod

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
    older MakerMods Lab versions left duplicates behind before dedup-at-registration."""
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
    from makermodslab.jobs import JobNotFoundError, JobRegistry

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
    from makermodslab.jobs import JobNotFoundError, JobRegistry

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
    from makermodslab.jobs import _flat_feature_dim

    assert _flat_feature_dim({"type": "STATE", "shape": [6]}) == 6
    assert _flat_feature_dim({"type": "STATE", "shape": [12]}) == 12
    assert _flat_feature_dim({"type": "ACTION", "shape": (12,)}) == 12


def test_flat_feature_dim_returns_none_for_missing_or_non_1d() -> None:
    from makermodslab.jobs import _flat_feature_dim

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

    from makermodslab.jobs import DatasetNotOnHubError, JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/local_only", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
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

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

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
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/on_hub", "status": "on_hub", "url": "u"},
        ),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", _fake_runner_factory),
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

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/maybe", policy_type="act")
    target = JobTarget(runner="hf_cloud", flavor="t4-small")

    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "job-xyz"
    fake_runner.hf_job_url.return_value = None

    with (
        patch(
            "makermodslab.datasets.get_hub_status",
            return_value={"repo_id": "user/maybe", "status": "unknown", "url": None},
        ),
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, target)

    assert record.runner == "hf_cloud"


def test_local_start_skips_hub_preflight(tmp_path) -> None:
    """A local run on a local-only dataset is fine — no Hub involved — so the
    preflight must not fire (get_hub_status is never consulted)."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    reg = JobRegistry(tmp_path / "root")
    cfg = TrainingRequest(dataset_repo_id="user/local_only", policy_type="act")

    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242

    with (
        patch("makermodslab.datasets.get_hub_status") as get_status,
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner),
    ):
        record = reg.start(cfg, JobTarget(runner="local"))

    get_status.assert_not_called()
    assert record.runner == "local"


# ── Imported-model card titles ───────────────────────────────────────────────
# The name an imported record is born with, and how the registry keeps two of
# them apart. The derivation rules themselves live in tests/test_naming.py.


def _hub_reg(monkeypatch, tmp_path, root="root"):
    """A registry whose Hub probe always reports a root-level checkpoint, so a
    hub import succeeds offline."""
    from makermodslab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: FakeApi())
    return JobRegistry(tmp_path / root)


def test_imported_name_is_the_derived_task_not_the_repo_id(monkeypatch, tmp_path) -> None:
    """No "Imported ·" prefix (the card's provenance chip already says it) and
    no namespace or policy token (the policy row says that) — just the task, so
    the title's pixels go to the one thing nothing else on the card states."""
    reg = _hub_reg(monkeypatch, tmp_path)
    rec = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")

    assert rec.name == "orange_box"
    assert "Imported" not in rec.name


def test_imported_name_from_a_local_dir(tmp_path) -> None:
    from makermodslab.jobs import JobRegistry

    model = tmp_path / "smolvla_me_orange_box_2026-08-03_12-53-30"
    _make_pretrained(model)
    reg = JobRegistry(tmp_path / "root")
    # No namespace on a path, so only the policy token and the timestamp go.
    assert reg.register_imported(str(model)).name == "me_orange_box"


def test_imported_name_keeps_a_community_repo_name(monkeypatch, tmp_path) -> None:
    """A repo with no generated timestamp was named by a human, so nothing but
    the namespace is dropped — the card falls back to a middle-ellipsized
    basename rather than guessing at structure that isn't there."""
    reg = _hub_reg(monkeypatch, tmp_path)
    assert reg.register_imported("lerobot/smolvla_base").name == "smolvla_base"


def test_explicit_import_name_wins_over_the_derivation(monkeypatch, tmp_path) -> None:
    """POST /jobs/import may carry a name. It's the user's, so neither the
    derivation nor the collision pass may overwrite it."""
    reg = _hub_reg(monkeypatch, tmp_path)
    rec = reg.register_imported(
        "makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30",
        name="Orange picker v2",
    )
    assert rec.name == "Orange picker v2"

    reg2 = _hub_reg(monkeypatch, tmp_path)  # reload from disk
    assert reg2.get(rec.id).name == "Orange picker v2"


def test_rename_alias_survives_the_collision_pass(monkeypatch, tmp_path) -> None:
    """A user-set display_name always wins on the card; re-deriving `name`
    underneath it must not disturb the alias."""
    reg = _hub_reg(monkeypatch, tmp_path)
    rec = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    reg.rename(rec.id, "Orange picker")

    reg2 = _hub_reg(monkeypatch, tmp_path)
    reloaded = reg2.get(rec.id)
    assert reloaded.display_name == "Orange picker"
    assert reloaded.name == "orange_box"


def test_two_imports_of_one_task_are_disambiguated(monkeypatch, tmp_path) -> None:
    """Both titles derive to "orange_box". BOTH cards get the timestamp back as
    a suffix — leaving the first bare would make it the ambiguous one."""
    reg = _hub_reg(monkeypatch, tmp_path)
    a = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    b = reg.register_imported("makermods/act_makermods_orange_box_2026-08-05_09-00-00")

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box (2026-08-03)"
    assert names[b.id] == "orange_box (2026-08-05)"


def test_same_day_imports_escalate_to_the_time(monkeypatch, tmp_path) -> None:
    reg = _hub_reg(monkeypatch, tmp_path)
    a = reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    b = reg.register_imported("makermods/act_makermods_orange_box_2026-08-03_18-04-00")

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box (2026-08-03 12:53)"
    assert names[b.id] == "orange_box (2026-08-03 18:04)"


def test_collision_suffixes_are_recomputed_not_accumulated(monkeypatch, tmp_path) -> None:
    """The pass is idempotent: it re-derives from the source every time, so a
    reload (or a third import) never stacks "(date) (date)" onto a title."""
    reg = _hub_reg(monkeypatch, tmp_path)
    reg.register_imported("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30")
    reg.register_imported("makermods/act_makermods_orange_box_2026-08-05_09-00-00")

    reg2 = _hub_reg(monkeypatch, tmp_path)
    reg3 = _hub_reg(monkeypatch, tmp_path)
    assert sorted(r.name for r in reg3.list(limit=100)) == sorted(r.name for r in reg2.list(limit=100))
    assert all(r.name.count("(") <= 1 for r in reg3.list(limit=100))


def test_legacy_imported_name_is_re_derived_on_load(monkeypatch, tmp_path) -> None:
    """Records written before titles were derived carry the old
    "Imported · <repo id>" name. Boot upgrades them in place, so the fix reaches
    the cards already in the user's library and not only the next import."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

    root = tmp_path / "root"
    job_dir = root / "smolvla_imported_2026-08-03_12-53-30"
    job_dir.mkdir(parents=True)
    legacy = JobRecord(
        id=job_dir.name,
        name="Imported · makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type="smolvla"),
        output_dir="",
        started_at=1.0,
        ended_at=1.0,
        runner="imported",
        hf_repo_id="makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30",
    )
    (job_dir / "job.json").write_text(legacy.model_dump_json(indent=2))

    reg = JobRegistry(root)
    assert reg.get(legacy.id).name == "orange_box"
    # Persisted, not just fixed in memory.
    assert _json.loads((job_dir / "job.json").read_text())["name"] == "orange_box"




def _typed_hub_reg(monkeypatch, tmp_path, policy_by_repo, root="root"):
    """`_hub_reg` plus a readable checkpoint config, so each import lands with a
    real `config.policy_type` — the field the card's Policy row renders, and the
    one the collision pass groups on."""
    reg = _hub_reg(monkeypatch, tmp_path, root=root)
    monkeypatch.setattr(
        "makermodslab.jobs._read_checkpoint_config",
        # A hub checkpoint's ref is "<repo_id>@root" (see _list_imported_hub).
        lambda ckpt: {"type": policy_by_repo[ckpt.ref.split("@", 1)[0]]},
    )
    return reg


def test_two_policies_of_one_task_are_not_disambiguated(monkeypatch, tmp_path) -> None:
    """Both derive to "orange_box", but one card says ACT and the other SmolVLA
    a line below the title — they are already told apart, so a date suffix would
    only crowd out the task name it is meant to clarify."""
    act_repo = "makermods/act_makermods_orange_box_2026-08-05_09-00-00"
    smolvla_repo = "makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30"
    reg = _typed_hub_reg(monkeypatch, tmp_path, {act_repo: "act", smolvla_repo: "smolvla"})
    a = reg.register_imported(smolvla_repo)
    b = reg.register_imported(act_repo)

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box"
    assert names[b.id] == "orange_box"


def test_two_imports_of_one_task_and_policy_are_still_disambiguated(monkeypatch, tmp_path) -> None:
    """Same task AND same policy: nothing on either card separates them, so the
    timestamp the title dropped comes back on both."""
    early = "makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30"
    late = "makermods/smolvla_makermods_orange_box_2026-08-05_09-00-00"
    reg = _typed_hub_reg(monkeypatch, tmp_path, {early: "smolvla", late: "smolvla"})
    a = reg.register_imported(early)
    b = reg.register_imported(late)

    names = {r.id: r.name for r in reg.list(limit=100)}
    assert names[a.id] == "orange_box (2026-08-03)"
    assert names[b.id] == "orange_box (2026-08-05)"


# ── Resume is only for a run that stopped short ──────────────────────────────
# A completed run's LR schedule is spent (SmolVLA's preset cosine-decays to a
# 2.5e-6 floor over a fixed 30k-step horizon), so a continuation trains at floor
# LR and the flat loss curve reads as convergence. The UI hides the button; this
# is the backend half, which also catches a direct API call. Blanket by
# decision — no per-policy exceptions.


def _hub_checkpoint_files(step_dir: str) -> list[str]:
    """The repo paths a complete cloud checkpoint publishes."""
    return [
        f"checkpoints/{step_dir}/pretrained_model/config.json",
        f"checkpoints/{step_dir}/pretrained_model/model.safetensors",
        f"checkpoints/{step_dir}/pretrained_model/train_config.json",
        f"checkpoints/{step_dir}/training_state/training_step.json",
        f"checkpoints/{step_dir}/training_state/optimizer_state.safetensors",
    ]


def _resumable_source(tmp_path, state: str, *, job_id: str = "src", steps: int = 200):
    """A local run record with one real, fully-saved checkpoint at step 100."""
    from makermodslab.jobs import JobRecord, JobRegistry
    from makermodslab.train import TrainingRequest

    run_dir = tmp_path / job_id / "run"
    run_dir.mkdir(parents=True)
    _make_checkpoint(run_dir, 100)
    reg = JobRegistry(tmp_path / "root")
    reg._records[job_id] = JobRecord(
        id=job_id,
        name="run",
        state=state,
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=steps),
        output_dir=str(run_dir),
        started_at=0.0,
        runner="local",
    )
    return reg


def _resume_request(job_id: str = "src"):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(
        dataset_repo_id="user/ds",
        policy_type="act",
        resume=True,
        resume_from_job_id=job_id,
    )


def test_start_refuses_to_resume_a_completed_run(tmp_path) -> None:
    """The point of the gate: a `done` source is refused with a 400-shaped
    ValueError that names fine-tuning as the way forward. No record created."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="already reached its step target"),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))

    assert [r.id for r in reg.list(limit=10)] == ["src"]


def test_start_refuses_to_resume_a_completed_cloud_run(tmp_path) -> None:
    """The refusal is on the source's STATE, not its runner, so it lands before
    the local/cloud branch and covers both."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "done")
    reg._records["src"].runner = "hf_cloud"
    reg._records["src"].hf_repo_id = "user/some-model"
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="already reached its step target"),
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))


@pytest.mark.parametrize("state", ["interrupted", "failed"])
def test_start_still_resumes_a_run_that_stopped_short(tmp_path, state) -> None:
    """The gate must not swallow the case resume exists for: a run that ended
    below its target still resumes, and still resolves its --config_path."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, state)
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    with patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="local"))

    assert record.config.resume is True
    assert record.config.config_path.endswith("train_config.json")
    assert "checkpoints/100" in record.config.config_path


# ── …and only on the runner it stopped on ────────────────────────────────────
# Cross-runner resume isn't implemented (F7) and both directions fail badly:
# cloud→local hands lerobot a --config_path that only ever existed inside the
# pod (MT4), local→cloud silently restarts from step 0 while recording itself as
# a resume (MT42). The form pins the Compute control; these cover the
# defense-in-depth half, which catches a direct API call that flips it anyway.
# Reuses the section's helpers above; a cloud parent is the same record with its
# runner/repo flipped, exactly as the done-source pair does it.


def _cloud_parent(reg, *, job_id: str = "src"):
    """Turn a `_resumable_source` record into a cloud run in place."""
    reg._records[job_id].runner = "hf_cloud"
    reg._records[job_id].hf_repo_id = "user/some-model"
    return reg


def test_start_refuses_a_local_parent_resumed_on_the_cloud(tmp_path, monkeypatch) -> None:
    """local parent → Cloud target (MT42). The refusal must name the parent's
    runner, and must leave no record behind."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _resumable_source(tmp_path, "interrupted")
    # The cloud dataset preflight sits ahead of the resume block; keep it happy
    # (and off the network) so the cross-runner refusal is what we observe.
    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "on_hub"})
    with (
        patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="Cross-runner resume isn't supported yet") as excinfo,
    ):
        reg.start(_resume_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert "Local" in str(excinfo.value)
    assert [r.id for r in reg.list(limit=10)] == ["src"]


def test_start_refuses_a_cloud_parent_resumed_locally(tmp_path, monkeypatch) -> None:
    """cloud parent → Local target (MT4). Refused BEFORE _resolve_cloud_resume
    would go to the Hub — the exploding api stand-in is the assertion that the
    guard lands first."""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    def _no_hub():
        raise AssertionError("cross-runner resume reached the Hub")

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", _no_hub)
    with (
        patch("makermodslab.jobs.LocalJobRunner", lambda *a, **k: MagicMock()),
        pytest.raises(ValueError, match="Cross-runner resume isn't supported yet") as excinfo,
    ):
        reg.start(_resume_request(), JobTarget(runner="local"))

    assert "Hugging Face Cloud" in str(excinfo.value)
    # Read the registry directly: `list` counts a cloud record's checkpoints,
    # which would itself call the stand-in above.
    assert list(reg._records) == ["src"]


def test_start_still_resumes_a_cloud_run_on_the_cloud(tmp_path, monkeypatch) -> None:
    """The other half of the gate: a same-runner cloud resume is untouched and
    still resolves the parent's Hub checkpoint. (local→local is covered by
    test_start_still_resumes_a_run_that_stopped_short above.)"""
    from unittest.mock import MagicMock, patch

    from makermodslab.jobs import JobTarget

    reg = _cloud_parent(_resumable_source(tmp_path, "interrupted"))
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(_hub_checkpoint_files("000100")),
    )
    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "on_hub"})
    fake_runner = MagicMock()
    fake_runner.hf_job_id.return_value = "hfjob-1"
    with patch("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: fake_runner):
        record = reg.start(_resume_request(), JobTarget(runner="hf_cloud", flavor="t4-small"))

    assert record.config.resume is True
    assert record.config.resume_from_hub_repo == "user/some-model"
    assert record.config.resume_from_hub_step == "000100"


# ---------------------------------------------------------------------------
# MT2 — fine-tuning a selected Hub step must actually train from THAT step.
# The resolver used to return `ref.split("@")[0]`, so a picked step silently
# became repo-ROOT weights. One rule now: a step-suffixed ref names the
# checkpoint, and whoever will run the trainer materializes it — host-side for a
# local run, in the container for a cloud one.
# ---------------------------------------------------------------------------


def _fake_snapshot(tmp_path: Path, seen: dict):
    """A snapshot_download stand-in that records its kwargs and lays down the
    tree the real one would (so the caller's path arithmetic is exercised)."""

    def _download(**kwargs):
        seen.update(kwargs)
        root = tmp_path / "snapshot"
        for pattern in kwargs.get("allow_patterns") or []:
            (root / pattern.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    return _download


def test_download_hub_checkpoint_ref_scopes_to_the_selected_step(monkeypatch, tmp_path) -> None:
    """Only that step's pretrained_model/ is pulled, and the returned path is
    that directory — the whole point being that lerobot cannot address a Hub
    sub-path itself."""
    from makermodslab.jobs import download_hub_checkpoint_ref

    seen: dict = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    out = download_hub_checkpoint_ref("user/repo@checkpoints/000500")

    assert seen["repo_id"] == "user/repo"
    assert seen["allow_patterns"] == ["checkpoints/000500/pretrained_model/*"]
    assert out.endswith("/checkpoints/000500/pretrained_model")


def test_download_hub_checkpoint_ref_root_skips_the_heavy_trees(monkeypatch, tmp_path) -> None:
    """A '@root' ref is the whole repo, minus the per-step snapshots and
    optimizer state — neither is needed to load the policy and both can be GBs."""
    from makermodslab.jobs import download_hub_checkpoint_ref

    seen: dict = {}
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    out = download_hub_checkpoint_ref("user/repo@root")

    assert seen["repo_id"] == "user/repo"
    assert seen["ignore_patterns"] == ["checkpoints/**", "training_state/**"]
    assert out == str(tmp_path / "snapshot")


def test_download_hub_checkpoint_ref_rejects_a_non_ref() -> None:
    from makermodslab.jobs import download_hub_checkpoint_ref

    with pytest.raises(ValueError, match="Unrecognised policy ref"):
        download_hub_checkpoint_ref("just-a-repo-id")


def test_localize_pretrained_path_passes_through_non_step_values(monkeypatch, tmp_path) -> None:
    """A local dir and a bare repo id are already loadable — lerobot resolves a
    repo root itself, so materializing one here would only duplicate its fetch."""
    from makermodslab.jobs import localize_pretrained_path

    def _no_downloads(**kwargs):
        raise AssertionError("nothing to materialize for a non-step value")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)
    assert localize_pretrained_path("user/some-model") == "user/some-model"
    assert localize_pretrained_path(str(tmp_path)) == str(tmp_path)


def test_localize_pretrained_path_reports_a_failed_download(monkeypatch) -> None:
    """A Hub failure becomes a 400-shaped ValueError naming the checkpoint,
    rather than a raw Hub exception surfacing as a 500."""
    from makermodslab.jobs import localize_pretrained_path

    def _boom(**kwargs):
        raise RuntimeError("hub down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)
    with pytest.raises(ValueError, match="Could not download the base checkpoint") as excinfo:
        localize_pretrained_path("user/repo@checkpoints/003000")
    assert "hub down" in str(excinfo.value)


def test_resolve_finetune_pretrained_path_keeps_the_selected_step(monkeypatch) -> None:
    """MT2 core: the step the user picked survives resolution. It used to be
    truncated to the repo id here, which is repo-ROOT weights."""
    from makermodslab.jobs import _resolve_finetune_pretrained_path

    files = _hub_checkpoint_files("001000") + _hub_checkpoint_files("005000")
    monkeypatch.setattr("makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(files))

    assert _resolve_finetune_pretrained_path(_cloud_record(), 1000) == "user/act_ds_2026@checkpoints/001000"
    # step=None still means "the latest", now equally un-truncated.
    assert _resolve_finetune_pretrained_path(_cloud_record(), None) == "user/act_ds_2026@checkpoints/005000"


def test_resolve_finetune_pretrained_path_root_ref_stays_a_repo_id(tmp_path) -> None:
    """An imported repo whose ROOT is the model needs no materialization —
    lerobot loads a repo id directly, so handing it the bare id avoids a
    pointless download."""
    from unittest.mock import patch

    from makermodslab.jobs import JobRecord, _resolve_finetune_pretrained_path
    from makermodslab.train import TrainingRequest

    record = JobRecord(
        id="imported-src",
        name="imported",
        state="done",
        config=TrainingRequest(dataset_repo_id="(imported)", policy_type="act"),
        output_dir="",
        started_at=0.0,
        runner="imported",
        hf_repo_id="user/flat_model",
    )
    with patch(
        "makermodslab.jobs.shared_hf_api",
        lambda: _FakeHubApi(["config.json", "model.safetensors"]),
    ):
        assert _resolve_finetune_pretrained_path(record, None) == "user/flat_model"


def _cloud_finetune_source(reg, repo_id: str = "user/act_ds_2026"):
    """A tracked cloud run in `reg` whose weights live on the Hub."""
    from makermodslab.jobs import JobRecord
    from makermodslab.train import TrainingRequest

    record = JobRecord(
        id="cloud-src",
        name="cloud src",
        state="done",
        config=TrainingRequest(dataset_repo_id="user/ds", policy_type="act", steps=10000),
        output_dir="",
        started_at=0.0,
        runner="hf_cloud",
        hf_repo_id=repo_id,
    )
    reg._records[record.id] = record
    return record


def test_finetune_start_local_materializes_the_selected_step(monkeypatch, tmp_path) -> None:
    """LOCAL target: the ref becomes a real directory on this machine before the
    trainer starts, and that directory is the SELECTED step."""
    from unittest.mock import MagicMock

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    seen: dict = {}
    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("003000"))
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(tmp_path, seen))

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)
    fake_runner = MagicMock()
    fake_runner.pid.return_value = 4242
    monkeypatch.setattr("makermodslab.jobs.LocalJobRunner", lambda *a, **k: fake_runner)

    record = reg.start(
        TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            finetune_from_job_id=source.id,
            finetune_from_step=3000,
        ),
        JobTarget(runner="local"),
    )

    resolved = record.config.policy_pretrained_path
    assert "@" not in resolved  # materialized, not a ref
    assert resolved.endswith("/checkpoints/003000/pretrained_model")
    assert seen["allow_patterns"] == ["checkpoints/003000/pretrained_model/*"]


def test_finetune_start_cloud_keeps_the_step_ref(monkeypatch, tmp_path) -> None:
    """CLOUD target: the ref is passed through untouched — a host path means
    nothing on the pod, so the container materializes the same ref itself. The
    host must NOT download the weights for a run that happens elsewhere."""
    from unittest.mock import MagicMock

    from makermodslab.jobs import JobRegistry, JobTarget
    from makermodslab.train import TrainingRequest

    def _no_downloads(**kwargs):
        raise AssertionError("the host must not download a cloud run's base weights")

    monkeypatch.setattr(
        "makermodslab.jobs.shared_hf_api", lambda: _FakeHubApi(_hub_checkpoint_files("003000"))
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_downloads)
    monkeypatch.setattr("makermodslab.datasets.get_hub_status", lambda repo_id: {"status": "on_hub"})
    monkeypatch.setattr("makermodslab.runners.hf_cloud.HfCloudJobRunner", lambda *a, **k: MagicMock())

    reg = JobRegistry(tmp_path / "root")
    source = _cloud_finetune_source(reg)

    record = reg.start(
        TrainingRequest(
            dataset_repo_id="user/ds",
            policy_type="act",
            finetune_from_job_id=source.id,
            finetune_from_step=3000,
        ),
        JobTarget(runner="hf_cloud", flavor="a10g-small"),
    )

    assert record.config.policy_pretrained_path == "user/act_ds_2026@checkpoints/003000"
