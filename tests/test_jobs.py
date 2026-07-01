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
"""Tests for lelab.jobs — parsers and Pydantic models. Does not exercise
LocalJobRunner.start() (see plan, "Discovered issue")."""

from __future__ import annotations

import json as _json
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
    from lelab.jobs import JobRecord
    from lelab.train import TrainingRequest

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
    from lelab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 5000)
    path = _resolve_resume_config_path(_record(out), 5000)
    assert path.endswith("checkpoints/5000/pretrained_model/train_config.json")


def test_resolve_resume_config_path_defaults_to_latest(tmp_path) -> None:
    from lelab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 1000)
    _make_checkpoint(out, 3000)
    path = _resolve_resume_config_path(_record(out), None)  # None ⇒ latest
    assert "checkpoints/3000/" in path


def test_resolve_resume_config_path_rejects_missing_training_state(tmp_path) -> None:
    from lelab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000, with_state=False)  # weights-only (e.g. imported)
    with pytest.raises(ValueError, match="training_state"):
        _resolve_resume_config_path(_record(out), 2000)


def test_resolve_resume_config_path_rejects_non_local(tmp_path) -> None:
    from lelab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="local"):
        _resolve_resume_config_path(_record(out, runner="hf_cloud"), 2000)


def test_resolve_resume_config_path_rejects_unknown_step(tmp_path) -> None:
    from lelab.jobs import _resolve_resume_config_path

    out = tmp_path / "run"
    _make_checkpoint(out, 2000)
    with pytest.raises(ValueError, match="no checkpoint at step 9999"):
        _resolve_resume_config_path(_record(out), 9999)


def test_extract_wandb_run_url_finds_canonical_url() -> None:
    from lelab.jobs import extract_wandb_run_url

    line = "wandb: \U0001f680 View run at https://wandb.ai/me/myproj/runs/abc123 trailing text"
    assert extract_wandb_run_url(line) == "https://wandb.ai/me/myproj/runs/abc123"


def test_extract_wandb_run_url_returns_none_when_absent() -> None:
    from lelab.jobs import extract_wandb_run_url

    assert extract_wandb_run_url("nothing here") is None
    assert extract_wandb_run_url("https://example.com/runs/abc") is None


def test_parse_duration_handles_mm_ss_and_hh_mm_ss() -> None:
    from lelab.jobs import _parse_duration

    assert _parse_duration("01:30") == 90
    assert _parse_duration("01:00:00") == 3600
    assert _parse_duration("?") is None
    assert _parse_duration("garbage") is None


def test_parse_metrics_into_extracts_loss_and_step() -> None:
    from lelab.jobs import TrainingMetrics, parse_metrics_into

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
    from lelab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  10%|██░| 1000/10000 [00:30<04:30, 3.2it/s]", m)
    parse_metrics_into("INFO ... step:1K smpl:8K loss:0.0077 grdn:0.9 lr:0.0001 ...", m)

    assert m.current_step == 1000  # kept from tqdm, not zeroed by "1K"
    assert m.current_loss == pytest.approx(0.0077)
    assert m.current_lr == pytest.approx(0.0001)


def test_parse_metrics_into_extracts_tqdm_progress() -> None:
    from lelab.jobs import TrainingMetrics, parse_metrics_into

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
    from lelab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into(
        "Training:  55%|█████| 55/100 [00:30<01:00, 2.0s/step]", m, resume_total=200
    )
    assert m.current_step == 155  # 200 - 100 + 55
    assert m.total_steps == 200


def test_parse_metrics_into_fresh_run_ignores_resume_rebase() -> None:
    """A fresh run passes resume_total=None; its bar is already the global step."""
    from lelab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("Training:  30%|███| 30/100 [00:30<01:00, 2.0s/step]", m)
    assert m.current_step == 30
    assert m.total_steps == 100


def test_read_metrics_history_stitches_resume_lineage(tmp_path) -> None:
    """A resumed run's curve is continuous across the whole lineage: the source
    run's points (0→100) are prepended to the resumed run's (150→200)."""
    from lelab.jobs import JobRecord, JobRegistry, LogLine, _job_log_path
    from lelab.train import TrainingRequest

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
    from lelab.jobs import TrainingMetrics, parse_metrics_into

    m = TrainingMetrics()
    parse_metrics_into("just a log line with no metrics", m)
    assert m.current_step == 0 or m.current_step is None  # accept either default


def test_log_line_round_trips_to_json() -> None:
    from lelab.jobs import LogLine

    line = LogLine(timestamp=1.5, message="hello")
    payload = line.model_dump_json()
    parsed = LogLine.model_validate_json(payload)
    assert parsed.timestamp == 1.5
    assert parsed.message == "hello"


def test_pid_alive_returns_false_for_unlikely_pid() -> None:
    from lelab.jobs import _pid_alive

    # DISCOVERED: os.kill(-1, 0) on macOS sends to process group and succeeds
    # (returns True), so we use a large PID that certainly does not exist.
    assert _pid_alive(999999999) is False


def test_hub_checkpoints_from_files_parses_tree() -> None:
    from lelab.jobs import _hub_checkpoints_from_files

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
    from lelab.jobs import _list_imported_local

    _make_pretrained(tmp_path)  # config.json at the root
    out = _list_imported_local(str(tmp_path))
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "local"
    assert out[0].ref == str(tmp_path.resolve())


def test_list_imported_local_checkpoints_tree(tmp_path) -> None:
    from lelab.jobs import _list_imported_local

    _make_pretrained(tmp_path / "checkpoints" / "000010" / "pretrained_model")
    out = _list_imported_local(str(tmp_path))
    assert [c.step for c in out] == [10]
    assert out[0].source == "local"
    assert out[0].ref.endswith("/checkpoints/000010/pretrained_model")


def test_list_imported_local_empty_when_no_model(tmp_path) -> None:
    from lelab.jobs import _list_imported_local

    assert _list_imported_local(str(tmp_path)) == []


def test_list_imported_hub_single_model() -> None:
    from lelab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors", "README.md"]

    out = _list_imported_hub(FakeApi(), "user/repo")
    assert len(out) == 1
    assert out[0].step == 0
    assert out[0].source == "hub"
    assert out[0].ref == "user/repo@root"


def test_list_imported_hub_prefers_checkpoints_tree() -> None:
    from lelab.jobs import _list_imported_hub

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
    from lelab.jobs import _list_imported_hub

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["README.md"]

    assert _list_imported_hub(FakeApi(), "user/repo") == []


def test_read_checkpoint_config_local_reads_config_json(tmp_path) -> None:
    from lelab.jobs import JobCheckpoint, _read_checkpoint_config

    (tmp_path / "config.json").write_text(_json.dumps({"type": "act"}))
    ckpt = JobCheckpoint(step=0, source="local", ref=str(tmp_path))
    assert _read_checkpoint_config(ckpt) == {"type": "act"}


def test_read_checkpoint_config_hub_root(monkeypatch, tmp_path) -> None:
    from lelab.jobs import JobCheckpoint, _read_checkpoint_config

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
    from lelab.jobs import JobCheckpoint, _read_checkpoint_config

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
    from lelab.jobs import JobRegistry

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
    from lelab.jobs import JobRegistry

    empty = tmp_path / "empty"
    empty.mkdir()
    reg = JobRegistry(tmp_path / "root")
    with pytest.raises(ValueError, match="No usable model"):
        reg.register_imported(str(empty))


def test_register_imported_hub_repo(monkeypatch, tmp_path) -> None:
    from lelab.jobs import JobRegistry

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["config.json", "model.safetensors"]

    monkeypatch.setattr("lelab.utils.hf_auth.shared_hf_api", lambda: FakeApi())
    reg = JobRegistry(tmp_path / "root")
    rec = reg.register_imported("user/some-model")

    assert rec.runner == "imported"
    assert rec.hf_repo_id == "user/some-model"
    assert rec.output_dir == ""
    cks = reg.list_checkpoints(rec.id)
    assert [c.ref for c in cks] == ["user/some-model@root"]
