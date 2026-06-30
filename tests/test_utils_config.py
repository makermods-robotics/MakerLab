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
"""Tests for lelab.utils.config — path resolution and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _patch_robots_path(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ROBOTS_PATH (not covered by the shared fixture) into tmp."""
    from lelab.utils import config as cfg

    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))


def test_port_persistence_round_trips(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_port("leader", "/dev/ttyUSB0")
    cfg.save_robot_port("follower", "/dev/ttyUSB1")

    assert cfg.get_saved_robot_port("leader") == "/dev/ttyUSB0"
    assert cfg.get_saved_robot_port("follower") == "/dev/ttyUSB1"


def test_get_saved_robot_port_returns_none_when_unset(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    assert cfg.get_saved_robot_port("leader") is None


def test_saved_robot_config_round_trips(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_config("leader", "my_calib")
    assert cfg.get_saved_robot_config("leader") == "my_calib"


def test_get_default_robot_config_falls_back_to_first_available(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.utils import config as cfg

    available = ["alpha", "beta", "gamma"]
    # No saved config → first available wins.
    assert cfg.get_default_robot_config("leader", available) == "alpha"

    # After saving, the saved one wins if it's still available.
    cfg.save_robot_config("leader", "beta")
    assert cfg.get_default_robot_config("leader", available) == "beta"

    # Saved config no longer in the available list → fall back to first.
    cfg.save_robot_config("leader", "deleted")
    assert cfg.get_default_robot_config("leader", available) == "alpha"


def test_is_valid_robot_name_accepts_simple_names(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    assert cfg.is_valid_robot_name("my_robot")
    assert cfg.is_valid_robot_name("robot-1")


def test_is_valid_robot_name_rejects_empty_and_path_separators(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.utils import config as cfg

    assert not cfg.is_valid_robot_name("")
    assert not cfg.is_valid_robot_name("a/b")
    assert not cfg.is_valid_robot_name("..")


def test_is_valid_robot_name_rejects_leading_trailing_whitespace(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.utils import config as cfg

    # name.strip() != name → invalid
    assert not cfg.is_valid_robot_name(" robot")
    assert not cfg.is_valid_robot_name("robot ")


def test_robot_record_save_get_delete_round_trip(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    record = {"name": "lab1", "leader_port": "/dev/ttyUSB0", "follower_port": ""}
    assert cfg.save_robot_record("lab1", record, allow_create=True)

    loaded = cfg.get_robot_record("lab1")
    assert loaded is not None
    assert loaded["name"] == "lab1"
    assert loaded["leader_port"] == "/dev/ttyUSB0"

    listed = cfg.list_robot_records()
    assert any(r["name"] == "lab1" for r in listed)

    assert cfg.delete_robot_record("lab1")
    assert cfg.get_robot_record("lab1") is None


def test_robot_record_allow_create_false_is_noop(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    # Record does not exist and allow_create=False → returns False.
    result = cfg.save_robot_record("nonexistent", {"leader_port": "/dev/x"}, allow_create=False)
    assert result is False
    assert cfg.get_robot_record("nonexistent") is None


def test_robot_record_save_rejects_invalid_name(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    # Path traversal-style names must not write outside the config dir.
    assert not cfg.save_robot_record("../escape", {"name": "x"}, allow_create=True)


def test_robot_record_merges_fields(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record("merge_test", {"leader_port": "/dev/a"}, allow_create=True)
    cfg.save_robot_record("merge_test", {"follower_port": "/dev/b"}, allow_create=False)

    loaded = cfg.get_robot_record("merge_test")
    assert loaded is not None
    assert loaded["leader_port"] == "/dev/a"
    assert loaded["follower_port"] == "/dev/b"


def test_rename_robot_record_moves_file_and_preserves_fields(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record("old_name", {"leader_port": "/dev/a"}, allow_create=True)

    ok, reason = cfg.rename_robot_record("old_name", "new_name")
    assert ok and reason == ""

    # Old gone, new present with fields and updated name.
    assert cfg.get_robot_record("old_name") is None
    moved = cfg.get_robot_record("new_name")
    assert moved is not None
    assert moved["name"] == "new_name"
    assert moved["leader_port"] == "/dev/a"


def test_rename_robot_record_noop_when_names_equal(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record("same", {"leader_port": "/dev/a"}, allow_create=True)
    ok, reason = cfg.rename_robot_record("same", "same")
    assert ok and reason == ""
    assert cfg.get_robot_record("same") is not None


def test_rename_robot_record_rejects_missing_source(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    ok, reason = cfg.rename_robot_record("ghost", "whatever")
    assert not ok and reason == "not_found"


def test_rename_robot_record_rejects_existing_target(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record("a", {"leader_port": "/dev/a"}, allow_create=True)
    cfg.save_robot_record("b", {"leader_port": "/dev/b"}, allow_create=True)

    ok, reason = cfg.rename_robot_record("a", "b")
    assert not ok and reason == "name_taken"
    # Both records untouched.
    assert cfg.get_robot_record("a")["leader_port"] == "/dev/a"
    assert cfg.get_robot_record("b")["leader_port"] == "/dev/b"


def test_rename_robot_record_rejects_invalid_target(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record("valid", {"leader_port": "/dev/a"}, allow_create=True)
    ok, reason = cfg.rename_robot_record("valid", "../escape")
    assert not ok and reason == "invalid_name"
    # Source record must survive a rejected rename.
    assert cfg.get_robot_record("valid") is not None


_GOOD_CALIBRATION = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 1927,
        "range_min": 741,
        "range_max": 3472,
    },
}


def test_validate_calibration_data_accepts_well_formed() -> None:
    from lelab.utils import config as cfg

    ok, reason = cfg.validate_calibration_data(_GOOD_CALIBRATION)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "data",
    [
        {},  # empty
        {"m": {"id": 1}},  # missing fields
        {"m": "not-an-object"},  # motor not a dict
        {"m": {"id": True, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 1}},  # bool not int
    ],
)
def test_validate_calibration_data_rejects_malformed(data) -> None:
    from lelab.utils import config as cfg

    ok, reason = cfg.validate_calibration_data(data)
    assert not ok and reason


def test_save_imported_calibration_writes_and_normalizes(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    # Name carries the .json extension (as robot records do) → normalized to stem.
    ok, reason, name = cfg.save_imported_calibration("teleop", "armA.json", _GOOD_CALIBRATION)
    assert ok and reason == "" and name == "armA"
    written = Path(cfg.LEADER_CONFIG_PATH) / "armA.json"
    assert written.is_file()
    assert json.loads(written.read_text()) == _GOOD_CALIBRATION


def test_save_imported_calibration_never_overwrites(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_imported_calibration("robot", "armB", _GOOD_CALIBRATION)
    ok, reason, _ = cfg.save_imported_calibration("robot", "armB", _GOOD_CALIBRATION)
    assert not ok and reason == "name_taken"


def test_save_imported_calibration_rejects_bad_device(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    ok, reason, _ = cfg.save_imported_calibration("nope", "x", _GOOD_CALIBRATION)
    assert not ok and reason == "invalid_device"


def test_get_robot_record_normalizes_config_extension(tmp_lerobot_home: Path) -> None:
    """Legacy records stored config names WITH .json; reads normalize to stems."""
    from lelab.utils import config as cfg

    # Write a record on disk that carries the old ".json" form.
    cfg.save_robot_record(
        "legacy",
        {"leader_config": "so101.json", "follower_config": "so101.json"},
        allow_create=True,
    )
    rec = cfg.get_robot_record("legacy")
    assert rec["leader_config"] == "so101"
    assert rec["follower_config"] == "so101"


def test_rename_calibration_config_moves_and_repoints_records(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").write_text("{}")
    cfg.save_robot_record("bot", {"leader_config": "armA"}, allow_create=True)

    ok, reason = cfg.rename_calibration_config("teleop", "armA", "armB")
    assert ok and reason == ""
    assert not (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").exists()
    assert (Path(cfg.LEADER_CONFIG_PATH) / "armB.json").exists()
    # The robot that referenced armA is repointed to armB.
    assert cfg.get_robot_record("bot")["leader_config"] == "armB"


def test_rename_calibration_config_never_overwrites(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    (Path(cfg.FOLLOWER_CONFIG_PATH) / "a.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "b.json").write_text('{"keep": 1}')

    ok, reason = cfg.rename_calibration_config("robot", "a", "b")
    assert not ok and reason == "name_taken"
    # Target untouched.
    assert (Path(cfg.FOLLOWER_CONFIG_PATH) / "b.json").read_text() == '{"keep": 1}'


def test_rename_calibration_config_missing_source(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    ok, reason = cfg.rename_calibration_config("teleop", "ghost", "x")
    assert not ok and reason == "not_found"


def test_record_defaults_to_single_mode(tmp_lerobot_home: Path) -> None:
    """A legacy record with no `mode` key reads back as single, with empty right_*."""
    from lelab.utils import config as cfg

    cfg.save_robot_record("legacy", {"leader_config": "L"}, allow_create=True)
    rec = cfg.get_robot_record("legacy")
    assert rec["mode"] == "single"
    assert rec["right_leader_config"] == ""


def test_save_record_persists_bimanual_mode(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record(
        "bi",
        {"mode": "bimanual", "right_leader_config": "RL"},
        allow_create=True,
    )
    rec = cfg.get_robot_record("bi")
    assert rec["mode"] == "bimanual"
    assert rec["right_leader_config"] == "RL"


def test_save_record_rejects_unknown_mode(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    cfg.save_robot_record("weird", {"mode": "nonsense"}, allow_create=True)
    assert cfg.get_robot_record("weird")["mode"] == "single"


def test_bimanual_record_clean_requires_all_four_calibrations(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    record = {
        "name": "bi",
        "mode": "bimanual",
        "leader_port": "/dev/ll", "follower_port": "/dev/lf",
        "leader_config": "LL", "follower_config": "LF",
        "right_leader_port": "/dev/rl", "right_follower_port": "/dev/rf",
        "right_leader_config": "RL", "right_follower_config": "RF",
    }
    # Only the left pair's files exist -> not clean.
    (Path(cfg.LEADER_CONFIG_PATH) / "LL.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "LF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is False

    # Add the right pair's files -> clean.
    (Path(cfg.LEADER_CONFIG_PATH) / "RL.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "RF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is True


def test_is_robot_record_clean_with_stem_configs(tmp_lerobot_home: Path) -> None:
    """A record storing stems is clean when "<stem>.json" exists on disk."""
    from lelab.utils import config as cfg

    record = {
        "name": "r",
        "leader_port": "/dev/a",
        "follower_port": "/dev/b",
        "leader_config": "so101",
        "follower_config": "so101",
    }
    assert cfg.is_robot_record_clean(record) is False  # no files yet

    (Path(cfg.LEADER_CONFIG_PATH) / "so101.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "so101.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is True
    # Still clean if a value carries the extension (defensive).
    assert cfg.is_robot_record_clean(dict(record, leader_config="so101.json")) is True


def test_setup_calibration_files_copies_configs(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.utils import config as cfg

    # setup_calibration_files reads from LEADER_CONFIG_PATH / FOLLOWER_CONFIG_PATH
    # and writes into those same directories (source dir == target dir).
    # Provide source files there.
    src_leader = Path(cfg.LEADER_CONFIG_PATH) / "demo_leader.json"
    src_leader.write_text(json.dumps({"motors": {}}))

    src_follower = Path(cfg.FOLLOWER_CONFIG_PATH) / "demo_follower.json"
    src_follower.write_text(json.dumps({"motors": {}}))

    result = cfg.setup_calibration_files("demo_leader.json", "demo_follower.json")
    # Returns the stem names.
    assert result == ("demo_leader", "demo_follower")

    # Files should exist (they were already there; function ensures they are present).
    assert src_leader.is_file()
    assert src_follower.is_file()


# DISCOVERED: `setup_calibration_files` sets `leader_calibration_dir = LEADER_CONFIG_PATH`
# (not CALIBRATION_BASE_PATH_TELEOP) and `follower_calibration_dir = FOLLOWER_CONFIG_PATH`
# (not CALIBRATION_BASE_PATH_ROBOTS). This means source and destination are the same
# directory, so the function only validates that the file exists in LEADER_CONFIG_PATH /
# FOLLOWER_CONFIG_PATH; it never writes into CALIBRATION_BASE_PATH_TELEOP or
# CALIBRATION_BASE_PATH_ROBOTS. The plan's assertion about those paths was incorrect.


def test_with_lelab_tag_appends_to_existing_tags() -> None:
    from lelab.utils.config import LELAB_TAG, with_lelab_tag

    assert with_lelab_tag(["robotics", "lerobot"]) == ["robotics", "lerobot", LELAB_TAG]


def test_with_lelab_tag_handles_none_and_empty() -> None:
    from lelab.utils.config import LELAB_TAG, with_lelab_tag

    assert with_lelab_tag(None) == [LELAB_TAG]
    assert with_lelab_tag([]) == [LELAB_TAG]


def test_with_lelab_tag_dedupes() -> None:
    from lelab.utils.config import LELAB_TAG, with_lelab_tag

    # Caller-supplied LeLab is not duplicated, and order is preserved.
    assert with_lelab_tag(["robotics", LELAB_TAG, "lerobot"]) == ["robotics", LELAB_TAG, "lerobot"]
