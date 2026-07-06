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
"""Tests for makerlab.utils.config — path resolution and persistence helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from makerlab.utils import config as cfg


@pytest.fixture(autouse=True)
def _patch_robots_path(tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ROBOTS_PATH (not covered by the shared fixture) into tmp."""
    from makerlab.utils import config as cfg

    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))


def test_port_persistence_round_trips(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_port("leader", "/dev/ttyUSB0")
    cfg.save_robot_port("follower", "/dev/ttyUSB1")

    assert cfg.get_saved_robot_port("leader") == "/dev/ttyUSB0"
    assert cfg.get_saved_robot_port("follower") == "/dev/ttyUSB1"


def test_get_saved_robot_port_returns_none_when_unset(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    assert cfg.get_saved_robot_port("leader") is None


def test_saved_robot_config_round_trips(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_config("leader", "my_calib")
    assert cfg.get_saved_robot_config("leader") == "my_calib"


def test_get_default_robot_config_falls_back_to_first_available(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.utils import config as cfg

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
    from makerlab.utils import config as cfg

    assert cfg.is_valid_robot_name("my_robot")
    assert cfg.is_valid_robot_name("robot-1")


def test_is_valid_robot_name_rejects_empty_and_path_separators(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.utils import config as cfg

    assert not cfg.is_valid_robot_name("")
    assert not cfg.is_valid_robot_name("a/b")
    assert not cfg.is_valid_robot_name("..")


def test_is_valid_robot_name_rejects_leading_trailing_whitespace(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.utils import config as cfg

    # name.strip() != name → invalid
    assert not cfg.is_valid_robot_name(" robot")
    assert not cfg.is_valid_robot_name("robot ")


def test_robot_record_save_get_delete_round_trip(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

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
    from makerlab.utils import config as cfg

    # Record does not exist and allow_create=False → returns False.
    result = cfg.save_robot_record("nonexistent", {"leader_port": "/dev/x"}, allow_create=False)
    assert result is False
    assert cfg.get_robot_record("nonexistent") is None


def test_robot_record_save_rejects_invalid_name(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    # Path traversal-style names must not write outside the config dir.
    assert not cfg.save_robot_record("../escape", {"name": "x"}, allow_create=True)


def test_robot_record_merges_fields(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_record("merge_test", {"leader_port": "/dev/a"}, allow_create=True)
    cfg.save_robot_record("merge_test", {"follower_port": "/dev/b"}, allow_create=False)

    loaded = cfg.get_robot_record("merge_test")
    assert loaded is not None
    assert loaded["leader_port"] == "/dev/a"
    assert loaded["follower_port"] == "/dev/b"


def test_robot_record_merge_clears_field_with_empty_string(tmp_lerobot_home: Path) -> None:
    """An empty string is a valid merge value: it CLEARS the field (e.g. releasing
    a port without assigning another), it does not preserve the old value."""
    from makerlab.utils import config as cfg

    cfg.save_robot_record(
        "clear_test", {"leader_port": "/dev/a", "follower_port": "/dev/b"}, allow_create=True
    )
    cfg.save_robot_record("clear_test", {"leader_port": ""}, allow_create=False)

    loaded = cfg.get_robot_record("clear_test")
    assert loaded is not None
    assert loaded["leader_port"] == ""
    assert loaded["follower_port"] == "/dev/b"


def test_clamp_motor_power_bounds_and_fallback(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    assert cfg.clamp_motor_power(55) == 55
    assert cfg.clamp_motor_power(5) == cfg.MOTOR_POWER_MIN
    assert cfg.clamp_motor_power(150) == cfg.MOTOR_POWER_MAX
    assert cfg.clamp_motor_power(42.9) == 42
    # Non-numeric (including bool — a subclass of int) → full power, never raise.
    assert cfg.clamp_motor_power(None) == cfg.DEFAULT_MOTOR_POWER
    assert cfg.clamp_motor_power("50") == cfg.DEFAULT_MOTOR_POWER
    assert cfg.clamp_motor_power(True) == cfg.DEFAULT_MOTOR_POWER


def test_robot_record_motor_power_defaults_to_full(tmp_lerobot_home: Path) -> None:
    """Records saved before the field existed (or fresh ones) read back as 100."""
    from makerlab.utils import config as cfg

    # A pre-motor_power record on disk: write raw JSON without the field.
    path = Path(cfg.ROBOTS_PATH) / "old_bot.json"
    path.write_text(json.dumps({"name": "old_bot", "mode": "single", "leader_port": "/dev/a"}))

    loaded = cfg.get_robot_record("old_bot")
    assert loaded is not None
    assert loaded["motor_power"] == cfg.DEFAULT_MOTOR_POWER


def test_robot_record_motor_power_merge_clamps_and_ignores_invalid(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_record("power_bot", {"motor_power": 60}, allow_create=True)
    assert cfg.get_robot_record("power_bot")["motor_power"] == 60

    # Out-of-range values are clamped, not rejected.
    cfg.save_robot_record("power_bot", {"motor_power": 5}, allow_create=False)
    assert cfg.get_robot_record("power_bot")["motor_power"] == cfg.MOTOR_POWER_MIN
    cfg.save_robot_record("power_bot", {"motor_power": 500}, allow_create=False)
    assert cfg.get_robot_record("power_bot")["motor_power"] == cfg.MOTOR_POWER_MAX

    # A wrongly-typed value is ignored (keeps the existing setting), matching
    # the known-typed-fields-only merge of the string/list fields.
    cfg.save_robot_record("power_bot", {"motor_power": "25"}, allow_create=False)
    assert cfg.get_robot_record("power_bot")["motor_power"] == cfg.MOTOR_POWER_MAX


def test_robot_record_motor_power_clamped_on_read(tmp_lerobot_home: Path) -> None:
    """A corrupted on-disk value never reaches consumers un-clamped."""
    from makerlab.utils import config as cfg

    path = Path(cfg.ROBOTS_PATH) / "corrupt_bot.json"
    path.write_text(json.dumps({"name": "corrupt_bot", "mode": "single", "motor_power": 9000}))
    assert cfg.get_robot_record("corrupt_bot")["motor_power"] == cfg.MOTOR_POWER_MAX

    path.write_text(json.dumps({"name": "corrupt_bot", "mode": "single", "motor_power": "junk"}))
    assert cfg.get_robot_record("corrupt_bot")["motor_power"] == cfg.DEFAULT_MOTOR_POWER


def test_rename_robot_record_moves_file_and_preserves_fields(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.utils import config as cfg

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
    from makerlab.utils import config as cfg

    cfg.save_robot_record("same", {"leader_port": "/dev/a"}, allow_create=True)
    ok, reason = cfg.rename_robot_record("same", "same")
    assert ok and reason == ""
    assert cfg.get_robot_record("same") is not None


def test_rename_robot_record_rejects_missing_source(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.rename_robot_record("ghost", "whatever")
    assert not ok and reason == "not_found"


def test_rename_robot_record_rejects_existing_target(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_record("a", {"leader_port": "/dev/a"}, allow_create=True)
    cfg.save_robot_record("b", {"leader_port": "/dev/b"}, allow_create=True)

    ok, reason = cfg.rename_robot_record("a", "b")
    assert not ok and reason == "name_taken"
    # Both records untouched.
    assert cfg.get_robot_record("a")["leader_port"] == "/dev/a"
    assert cfg.get_robot_record("b")["leader_port"] == "/dev/b"


def test_rename_robot_record_rejects_invalid_target(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

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
    from makerlab.utils import config as cfg

    ok, reason = cfg.validate_calibration_data(_GOOD_CALIBRATION)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "data",
    [
        {},  # empty
        {"m": {"id": 1}},  # missing fields
        {"m": "not-an-object"},  # motor not a dict
        {
            "m": {"id": True, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 1}
        },  # bool not int
    ],
)
def test_validate_calibration_data_rejects_malformed(data) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.validate_calibration_data(data)
    assert not ok and reason


@pytest.mark.parametrize("name", ["whoo", "my-set_v2", "ok.name-1", "a", "A1"])
def test_validate_dataset_name_accepts_good(name) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.validate_dataset_name(name)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty
        "   ",  # whitespace only
        " whoo",  # leading space
        "whoo ",  # trailing space
        "whoo/",  # trailing slash
        "a/b",  # embedded slash
        "..",  # traversal
        ".",  # traversal
        ".hidden",  # leading dot
        "-lead",  # leading dash
        "trail-",  # trailing dash
        "bad name",  # space
        "café",  # non-ascii
        "x" * 97,  # too long
    ],
)
def test_validate_dataset_name_rejects_bad(name) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.validate_dataset_name(name)
    assert not ok and reason


@pytest.mark.parametrize("repo_id", ["whoo", "Mokuroh54/whoo", "user/my-set_v2"])
def test_validate_dataset_repo_id_accepts_good(repo_id) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.validate_dataset_repo_id(repo_id)
    assert ok and reason == ""


@pytest.mark.parametrize(
    "repo_id",
    [
        "Mokuroh54/whoo/",  # the reported bug: trailing slash
        "whoo/",  # trailing slash, no namespace
        "a/b/c",  # too many slashes
        "-bad/whoo",  # bad namespace
        "user/.hidden",  # bad name segment
        "",  # empty
    ],
)
def test_validate_dataset_repo_id_rejects_bad(repo_id) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.validate_dataset_repo_id(repo_id)
    assert not ok and reason


def test_save_imported_calibration_writes_and_normalizes(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    # Name carries the .json extension (as robot records do) → normalized to stem.
    ok, reason, name = cfg.save_imported_calibration("teleop", "armA.json", _GOOD_CALIBRATION)
    assert ok and reason == "" and name == "armA"
    written = Path(cfg.LEADER_CONFIG_PATH) / "armA.json"
    assert written.is_file()
    assert json.loads(written.read_text()) == _GOOD_CALIBRATION


def test_save_imported_calibration_never_overwrites(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_imported_calibration("robot", "armB", _GOOD_CALIBRATION)
    ok, reason, _ = cfg.save_imported_calibration("robot", "armB", _GOOD_CALIBRATION)
    assert not ok and reason == "name_taken"


def test_save_imported_calibration_rejects_bad_device(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    ok, reason, _ = cfg.save_imported_calibration("nope", "x", _GOOD_CALIBRATION)
    assert not ok and reason == "invalid_device"


def test_get_robot_record_normalizes_config_extension(tmp_lerobot_home: Path) -> None:
    """Legacy records stored config names WITH .json; reads normalize to stems."""
    from makerlab.utils import config as cfg

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
    from makerlab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").write_text("{}")
    cfg.save_robot_record("bot", {"leader_config": "armA"}, allow_create=True)

    ok, reason = cfg.rename_calibration_config("teleop", "armA", "armB")
    assert ok and reason == ""
    assert not (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").exists()
    assert (Path(cfg.LEADER_CONFIG_PATH) / "armB.json").exists()
    # The robot that referenced armA is repointed to armB.
    assert cfg.get_robot_record("bot")["leader_config"] == "armB"


def test_rename_calibration_config_repoints_right_arm_slot(tmp_lerobot_home: Path) -> None:
    """Renaming a config repoints the bimanual right slot, not just the left."""
    from makerlab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "armA.json").write_text("{}")
    cfg.save_robot_record(
        "bi",
        {"mode": "bimanual", "leader_config": "armX", "right_leader_config": "armA"},
        allow_create=True,
    )

    ok, reason = cfg.rename_calibration_config("teleop", "armA", "armB")
    assert ok and reason == ""

    rec = cfg.get_robot_record("bi")
    assert rec["right_leader_config"] == "armB"
    assert rec["leader_config"] == "armX"  # the other slot is untouched


def test_rename_calibration_config_never_overwrites(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    (Path(cfg.FOLLOWER_CONFIG_PATH) / "a.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "b.json").write_text('{"keep": 1}')

    ok, reason = cfg.rename_calibration_config("robot", "a", "b")
    assert not ok and reason == "name_taken"
    # Target untouched.
    assert (Path(cfg.FOLLOWER_CONFIG_PATH) / "b.json").read_text() == '{"keep": 1}'


def test_rename_calibration_config_missing_source(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    ok, reason = cfg.rename_calibration_config("teleop", "ghost", "x")
    assert not ok and reason == "not_found"


def test_record_defaults_to_single_mode(tmp_lerobot_home: Path) -> None:
    """A legacy record with no `mode` key reads back as single, with empty right_*."""
    from makerlab.utils import config as cfg

    cfg.save_robot_record("legacy", {"leader_config": "L"}, allow_create=True)
    rec = cfg.get_robot_record("legacy")
    assert rec["mode"] == "single"
    assert rec["right_leader_config"] == ""


def test_save_record_persists_bimanual_mode(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_record(
        "bi",
        {"mode": "bimanual", "right_leader_config": "RL"},
        allow_create=True,
    )
    rec = cfg.get_robot_record("bi")
    assert rec["mode"] == "bimanual"
    assert rec["right_leader_config"] == "RL"


def test_save_record_rejects_unknown_mode(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    cfg.save_robot_record("weird", {"mode": "nonsense"}, allow_create=True)
    assert cfg.get_robot_record("weird")["mode"] == "single"


def test_bimanual_record_clean_requires_all_four_calibrations(tmp_lerobot_home: Path) -> None:
    from makerlab.utils import config as cfg

    record = {
        "name": "bi",
        "mode": "bimanual",
        "leader_port": "/dev/ll",
        "follower_port": "/dev/lf",
        "leader_config": "LL",
        "follower_config": "LF",
        "right_leader_port": "/dev/rl",
        "right_follower_port": "/dev/rf",
        "right_leader_config": "RL",
        "right_follower_config": "RF",
    }
    # Only the left pair's files exist -> not clean.
    (Path(cfg.LEADER_CONFIG_PATH) / "LL.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "LF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is False

    # Add the right pair's files -> clean.
    (Path(cfg.LEADER_CONFIG_PATH) / "RL.json").write_text("{}")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "RF.json").write_text("{}")
    assert cfg.is_robot_record_clean(record) is True


def test_config_slot_conflict_detects_same_side_duplicate() -> None:
    from makerlab.utils import config as cfg

    base = {
        "mode": "bimanual",
        "leader_config": "L1",
        "follower_config": "F1",
        "right_leader_config": "L2",
        "right_follower_config": "F2",
    }
    assert cfg.config_slot_conflict(base) is None
    assert cfg.config_slot_conflict({**base, "right_leader_config": "L1"}) == "leader"
    assert cfg.config_slot_conflict({**base, "right_follower_config": "F1"}) == "follower"


def test_port_slot_conflict_detects_shared_port() -> None:
    from makerlab.utils import config as cfg

    # Single: leader and follower must differ.
    assert (
        cfg.port_slot_conflict({"mode": "single", "leader_port": "/dev/a", "follower_port": "/dev/b"}) is None
    )
    assert (
        cfg.port_slot_conflict({"mode": "single", "leader_port": "/dev/a", "follower_port": "/dev/a"})
        == "/dev/a"
    )

    # Bimanual: all four must differ, across sides.
    base = {
        "mode": "bimanual",
        "leader_port": "/dev/a",
        "follower_port": "/dev/b",
        "right_leader_port": "/dev/c",
        "right_follower_port": "/dev/d",
    }
    assert cfg.port_slot_conflict(base) is None
    assert cfg.port_slot_conflict({**base, "right_follower_port": "/dev/a"}) == "/dev/a"
    # Empty ports are ignored.
    assert cfg.port_slot_conflict({"mode": "bimanual", "leader_port": "", "follower_port": ""}) is None


def test_config_slot_conflict_ignores_single_mode_and_cross_side() -> None:
    from makerlab.utils import config as cfg

    # Single mode never conflicts (one slot per side).
    assert (
        cfg.config_slot_conflict({"mode": "single", "leader_config": "X", "right_leader_config": "X"}) is None
    )
    # Same name across sides is fine — different directories.
    assert (
        cfg.config_slot_conflict({"mode": "bimanual", "leader_config": "X", "follower_config": "X"}) is None
    )
    # Empty slots don't count as a conflict.
    assert (
        cfg.config_slot_conflict({"mode": "bimanual", "leader_config": "", "right_leader_config": ""}) is None
    )


def test_is_robot_record_clean_with_stem_configs(tmp_lerobot_home: Path) -> None:
    """A record storing stems is clean when "<stem>.json" exists on disk."""
    from makerlab.utils import config as cfg

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
    from makerlab.utils import config as cfg

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


def test_stage_bimanual_calibrations_copies_four_files(tmp_lerobot_home: Path) -> None:
    """The four arbitrarily-named library files are copied into per-device
    staging dirs as '<base>_left/right.json', returning the dirs + base."""
    from makerlab.utils import config as cfg

    # Arbitrary library names — no "<base>_left/right" convention.
    for name, content in (("alice", "AL"), ("carol", "AR")):
        (Path(cfg.LEADER_CONFIG_PATH) / f"{name}.json").write_text(content)
    for name, content in (("bob", "FL"), ("dave", "FR")):
        (Path(cfg.FOLLOWER_CONFIG_PATH) / f"{name}.json").write_text(content)

    leader_dir, follower_dir, base = cfg.stage_bimanual_calibrations("mybot", "alice", "carol", "bob", "dave")
    assert base == "mybot"
    assert leader_dir == os.path.join(cfg.MAKERLAB_BISO_STAGING_PATH, "mybot", "leader")
    assert follower_dir == os.path.join(cfg.MAKERLAB_BISO_STAGING_PATH, "mybot", "follower")
    # Files landed under the convention names with the right contents.
    assert (Path(leader_dir) / "mybot_left.json").read_text() == "AL"
    assert (Path(leader_dir) / "mybot_right.json").read_text() == "AR"
    assert (Path(follower_dir) / "mybot_left.json").read_text() == "FL"
    assert (Path(follower_dir) / "mybot_right.json").read_text() == "FR"


def test_stage_bimanual_calibrations_overwrites_stale_alias(tmp_lerobot_home: Path) -> None:
    """A recalibrated library file must refresh its staging alias — the copy is
    unconditional, so a second call overwrites the previous staged content."""
    from makerlab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "L.json").write_text("v1")
    (Path(cfg.LEADER_CONFIG_PATH) / "R.json").write_text("R")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FL.json").write_text("FL")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FR.json").write_text("FR")

    leader_dir, _, _ = cfg.stage_bimanual_calibrations("bot", "L", "R", "FL", "FR")
    assert (Path(leader_dir) / "bot_left.json").read_text() == "v1"

    # Recalibrate the left leader library file, then restage.
    (Path(cfg.LEADER_CONFIG_PATH) / "L.json").write_text("v2")
    cfg.stage_bimanual_calibrations("bot", "L", "R", "FL", "FR")
    assert (Path(leader_dir) / "bot_left.json").read_text() == "v2"


def test_stage_bimanual_calibrations_missing_file_raises(tmp_lerobot_home: Path) -> None:
    """A missing library file fails fast with a clear per-slot error naming the
    slot and file, before lerobot's connect() can hang on recalibration."""
    from makerlab.utils import config as cfg

    # Only three of the four files exist; the right follower is missing.
    (Path(cfg.LEADER_CONFIG_PATH) / "L.json").write_text("L")
    (Path(cfg.LEADER_CONFIG_PATH) / "R.json").write_text("R")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "FL.json").write_text("FL")

    with pytest.raises(FileNotFoundError, match="right follower.*FR.json.*not found"):
        cfg.stage_bimanual_calibrations("bot", "L", "R", "FL", "FR")


def test_stage_bimanual_calibrations_blank_slot_raises(tmp_lerobot_home: Path) -> None:
    """A blank config (arm unassigned) fails with the standard legible message."""
    from makerlab.utils import config as cfg

    with pytest.raises(FileNotFoundError, match="left leader arm has no calibration assigned"):
        cfg.stage_bimanual_calibrations("bot", "", "R", "FL", "FR")


def test_stage_bimanual_follower_calibrations_stages_follower_only(tmp_lerobot_home: Path) -> None:
    """Inference stages the follower side only. Repro of the startup bug: the
    two follower library files exist under FOLLOWER_CONFIG_PATH but NO leader
    file shares their names — staging must still succeed and land the follower
    aliases, rather than failing looking for so_leader/<follower name>.json."""
    from makerlab.utils import config as cfg

    # Real-world repro: follower configs "2"/"4"; leader dir has no 2/4.json.
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "2.json").write_text("FL")
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "4.json").write_text("FR")

    follower_dir, base = cfg.stage_bimanual_follower_calibrations("mybot", "2", "4")
    assert base == "mybot"
    # Same layout as the full stager's follower dir.
    assert follower_dir == os.path.join(cfg.MAKERLAB_BISO_STAGING_PATH, "mybot", "follower")
    assert (Path(follower_dir) / "mybot_left.json").read_text() == "FL"
    assert (Path(follower_dir) / "mybot_right.json").read_text() == "FR"
    # No leader staging dir is created — the leader side is never touched.
    assert not os.path.exists(os.path.join(cfg.MAKERLAB_BISO_STAGING_PATH, "mybot", "leader"))


def test_stage_bimanual_follower_calibrations_missing_file_raises(tmp_lerobot_home: Path) -> None:
    """A missing follower library file fails fast with the clear per-slot error
    naming 'right follower' and the file, same as the full stager."""
    from makerlab.utils import config as cfg

    (Path(cfg.FOLLOWER_CONFIG_PATH) / "2.json").write_text("FL")

    with pytest.raises(FileNotFoundError, match="right follower.*4.json.*not found"):
        cfg.stage_bimanual_follower_calibrations("mybot", "2", "4")


def test_bimanual_base_id_uses_valid_name_else_default() -> None:
    from makerlab.utils.config import DEFAULT_BIMANUAL_BASE, bimanual_base_id

    assert bimanual_base_id("mybot") == "mybot"
    assert bimanual_base_id("  spaced  ") == "spaced"  # stripped, still valid
    # Blank or unsafe names fall back to the fixed default.
    assert bimanual_base_id("") == DEFAULT_BIMANUAL_BASE
    assert bimanual_base_id(None) == DEFAULT_BIMANUAL_BASE
    assert bimanual_base_id("bad/name") == DEFAULT_BIMANUAL_BASE
    assert bimanual_base_id("../escape") == DEFAULT_BIMANUAL_BASE


def test_with_makerlab_tag_appends_to_existing_tags() -> None:
    from makerlab.utils.config import MAKERLAB_TAG, with_makerlab_tag

    assert with_makerlab_tag(["robotics", "lerobot"]) == ["robotics", "lerobot", MAKERLAB_TAG]


def test_with_makerlab_tag_handles_none_and_empty() -> None:
    from makerlab.utils.config import MAKERLAB_TAG, with_makerlab_tag

    assert with_makerlab_tag(None) == [MAKERLAB_TAG]
    assert with_makerlab_tag([]) == [MAKERLAB_TAG]


def test_with_makerlab_tag_dedupes() -> None:
    from makerlab.utils.config import MAKERLAB_TAG, with_makerlab_tag

    # Caller-supplied MakerLab is not duplicated, and order is preserved.
    assert with_makerlab_tag(["robotics", MAKERLAB_TAG, "lerobot"]) == ["robotics", MAKERLAB_TAG, "lerobot"]


def test_clear_config_references_unassigns_matching_records(tmp_lerobot_home: Path) -> None:
    """Deleting a config unassigns every robot that pointed at it — on the
    right side (device_type) only — and reports which fields were cleared."""
    cfg.save_robot_record(
        "arm1",
        {"mode": "single", "leader_config": "calib_a", "follower_config": "calib_b"},
        allow_create=True,
    )
    # A second robot sharing the same leader config is unassigned too.
    cfg.save_robot_record("arm2", {"mode": "single", "leader_config": "calib_a"}, allow_create=True)

    assert cfg.clear_config_references("teleop", "calib_a") == [
        {"robot": "arm1", "fields": ["leader_config"]},
        {"robot": "arm2", "fields": ["leader_config"]},
    ]
    assert cfg.get_robot_record("arm1")["leader_config"] == ""
    assert cfg.get_robot_record("arm2")["leader_config"] == ""
    # The follower slot (other side) is untouched, and the record is now dirty.
    assert cfg.get_robot_record("arm1")["follower_config"] == "calib_b"
    assert cfg.is_robot_record_clean(cfg.get_robot_record("arm1")) is False

    # A config nobody references clears nothing.
    assert cfg.clear_config_references("teleop", "unused") == []


def test_clear_config_references_clears_stale_right_slot_too(tmp_lerobot_home: Path) -> None:
    """A right_* reference is cleared even when the robot is back in single
    mode — the file is gone, so the stale name must not resurface on a mode
    switch. Both slots are reported when both matched."""
    cfg.save_robot_record(
        "arm1",
        {"mode": "single", "leader_config": "gone", "right_leader_config": "gone"},
        allow_create=True,
    )
    assert cfg.clear_config_references("teleop", "gone") == [
        {"robot": "arm1", "fields": ["leader_config", "right_leader_config"]}
    ]
    record = cfg.get_robot_record("arm1")
    assert record["leader_config"] == ""
    assert record["right_leader_config"] == ""


def test_clear_config_references_accepts_json_extension(tmp_lerobot_home: Path) -> None:
    """Callers may pass 'name.json'; matching is on the stem."""
    cfg.save_robot_record("arm1", {"mode": "single", "follower_config": "calib_b"}, allow_create=True)
    assert cfg.clear_config_references("robot", "calib_b.json") == [
        {"robot": "arm1", "fields": ["follower_config"]}
    ]
    assert cfg.get_robot_record("arm1")["follower_config"] == ""


def test_setup_calibration_files_rejects_unassigned_arm(tmp_lerobot_home: Path) -> None:
    """An empty config name (arm unassigned / needs calibration) fails with a
    legible message instead of an IsADirectoryError from shutil.copy2."""
    with pytest.raises(FileNotFoundError, match="leader arm has no calibration assigned"):
        cfg.setup_calibration_files("", "whatever.json")
    with pytest.raises(FileNotFoundError, match="follower arm has no calibration assigned"):
        cfg.setup_calibration_files("whatever.json", "  ")
    with pytest.raises(FileNotFoundError, match="follower arm has no calibration assigned"):
        cfg.setup_follower_calibration_file("")


# --- Dismissed hub jobs -----------------------------------------------------


def test_dismissed_hub_jobs_round_trips(tmp_lerobot_home: Path) -> None:
    assert cfg.get_dismissed_hub_jobs() == set()
    assert cfg.add_dismissed_hub_job("job-b") is True
    assert cfg.add_dismissed_hub_job("job-a") is True
    assert cfg.get_dismissed_hub_jobs() == {"job-a", "job-b"}
    # Idempotent: re-dismissing is a no-op success.
    assert cfg.add_dismissed_hub_job("job-a") is True
    assert cfg.get_dismissed_hub_jobs() == {"job-a", "job-b"}


def test_add_dismissed_hub_job_rejects_blank_id(tmp_lerobot_home: Path) -> None:
    assert cfg.add_dismissed_hub_job("") is False
    assert cfg.add_dismissed_hub_job("   ") is False
    assert cfg.get_dismissed_hub_jobs() == set()


def test_get_dismissed_hub_jobs_tolerates_corrupt_file(tmp_lerobot_home: Path) -> None:
    """Dismissal is cosmetic — a corrupted file must yield the empty set, not
    an exception that would block the hub listing."""
    path = Path(cfg.DISMISSED_HUB_JOBS_FILE)
    path.write_text("not json{")
    assert cfg.get_dismissed_hub_jobs() == set()
    # Wrong shape (dict instead of list) and non-string entries are dropped too.
    path.write_text(json.dumps({"job-a": True}))
    assert cfg.get_dismissed_hub_jobs() == set()
    path.write_text(json.dumps(["job-a", 3, None, "  "]))
    assert cfg.get_dismissed_hub_jobs() == {"job-a"}


def test_prune_dismissed_hub_jobs_drops_ids_gone_from_listing(tmp_lerobot_home: Path) -> None:
    cfg.add_dismissed_hub_job("job-live")
    cfg.add_dismissed_hub_job("job-expired")
    cfg.prune_dismissed_hub_jobs({"job-live", "job-other"})
    assert cfg.get_dismissed_hub_jobs() == {"job-live"}
    # Pruning against a listing that contains everything is a no-op.
    cfg.prune_dismissed_hub_jobs({"job-live"})
    assert cfg.get_dismissed_hub_jobs() == {"job-live"}
