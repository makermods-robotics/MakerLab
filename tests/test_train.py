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
"""Tests for lelab.train — request schema and CLI builder."""

from __future__ import annotations

import pytest


def _arg_value(cmd: list[str], flag: str) -> str:
    """Return the value passed to `--flag`. Fails the test if absent."""
    assert flag in cmd, f"{flag} missing from {cmd}"
    return cmd[cmd.index(flag) + 1]


def test_minimal_request_yields_well_formed_argv() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert _arg_value(cmd, "--dataset.repo_id") == "lerobot/pusht"
    assert _arg_value(cmd, "--policy.type") == "act"
    assert _arg_value(cmd, "--steps") == "10000"
    assert _arg_value(cmd, "--output_dir") == "/tmp/out"


def test_resume_request_emits_minimal_argv() -> None:
    """On resume, lerobot reconstructs the run from config_path, so the builder
    must NOT re-pass --dataset.* / --policy.type (they'd fight the loaded
    config) and must pass the resume essentials plus the overridable knobs."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="lerobot/pusht",
        resume=True,
        config_path="/runs/abc/checkpoints/5000/pretrained_model/train_config.json",
        steps=20000,
    )
    cmd = build_training_command(req, output_dir="/tmp/new")

    # config_path MUST be the "--config_path=<path>" form: lerobot's own
    # pre-parser ignores the space-separated form.
    cfg_args = [a for a in cmd if a.startswith("--config_path=")]
    assert cfg_args == [
        "--config_path=/runs/abc/checkpoints/5000/pretrained_model/train_config.json"
    ]
    assert "--config_path" not in cmd  # not the two-token form
    assert _arg_value(cmd, "--resume") == "true"
    assert _arg_value(cmd, "--output_dir") == "/tmp/new"
    assert _arg_value(cmd, "--steps") == "20000"
    # Inherited from the checkpoint — must not be re-specified on the CLI.
    assert "--dataset.repo_id" not in cmd
    assert "--policy.type" not in cmd


def test_optional_dataset_fields_only_present_when_set() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, "/tmp/out")
    assert "--dataset.revision" not in cmd
    assert "--dataset.root" not in cmd
    assert "--dataset.episodes" not in cmd

    req2 = TrainingRequest(
        dataset_repo_id="lerobot/pusht",
        dataset_revision="v2",
        dataset_root="/data",
        dataset_episodes=[0, 1, 2],
    )
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--dataset.revision") == "v2"
    assert _arg_value(cmd2, "--dataset.root") == "/data"
    # `--dataset.episodes` is followed by 3 string-encoded ints.
    idx = cmd2.index("--dataset.episodes")
    assert cmd2[idx + 1 : idx + 4] == ["0", "1", "2"]


def test_wandb_block_only_serialized_when_enabled() -> None:
    from lelab.train import TrainingRequest, build_training_command

    off = build_training_command(TrainingRequest(dataset_repo_id="x", wandb_enable=False), "/tmp/out")
    assert _arg_value(off, "--wandb.enable") == "false"
    assert "--wandb.project" not in off

    on = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            wandb_enable=True,
            wandb_project="proj",
            wandb_entity="me",
            wandb_run_id="abc",
        ),
        "/tmp/out",
    )
    assert _arg_value(on, "--wandb.enable") == "true"
    assert _arg_value(on, "--wandb.project") == "proj"
    assert _arg_value(on, "--wandb.entity") == "me"
    assert _arg_value(on, "--wandb.run_id") == "abc"


def test_push_to_hub_emits_repo_id_only_when_enabled() -> None:
    from lelab.train import TrainingRequest, build_training_command

    off = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=False, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _arg_value(off, "--policy.push_to_hub") == "false"
    assert "--policy.repo_id" not in off

    on = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=True, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _arg_value(on, "--policy.push_to_hub") == "true"
    assert _arg_value(on, "--policy.repo_id") == "me/x"


def test_seed_omitted_when_none() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="x", seed=None)
    cmd = build_training_command(req, "/tmp/out")
    assert "--seed" not in cmd

    req2 = TrainingRequest(dataset_repo_id="x", seed=42)
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--seed") == "42"


def test_explicit_device_passes_through() -> None:
    """A concrete device (persisted by an older config) passes through
    unchanged for backward compatibility."""
    from lelab.train import TrainingRequest, build_training_command

    cmd = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_device="cuda"), "/tmp/out"
    )
    assert _arg_value(cmd, "--policy.device") == "cuda"

    cmd_cpu = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_device="cpu"), "/tmp/out"
    )
    assert _arg_value(cmd_cpu, "--policy.device") == "cpu"


def test_auto_device_resolves_to_concrete_backend(monkeypatch) -> None:
    """The default "auto" resolves to a real backend so the logged config is
    truthful. Resolution is made deterministic here via monkeypatch."""
    import torch

    from lelab.train import TrainingRequest, build_training_command

    # No GPU available -> cpu.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cmd = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_device="auto"), "/tmp/out"
    )
    assert _arg_value(cmd, "--policy.device") == "cpu"

    # CUDA available -> cuda.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cmd_cuda = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_device="auto"), "/tmp/out"
    )
    assert _arg_value(cmd_cuda, "--policy.device") == "cuda"


def test_default_device_is_auto_and_resolved(monkeypatch) -> None:
    """The request default is "auto" (not "cuda"); build resolves it to a
    concrete backend rather than emitting "auto"."""
    import torch

    from lelab.train import TrainingRequest, build_training_command

    assert TrainingRequest(dataset_repo_id="x").policy_device == "auto"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    cmd = build_training_command(TrainingRequest(dataset_repo_id="x"), "/tmp/out")
    assert _arg_value(cmd, "--policy.device") == "mps"


def test_training_request_validates_required_field() -> None:
    from pydantic import ValidationError

    from lelab.train import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest()  # dataset_repo_id is required
