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
"""Shared leader/follower config-object assembly for SO-101 arms.

Teleoperation and recording both build the same pair of lerobot config
objects — a follower ``robot_config`` and a leader ``teleop_config`` — from
a request that carries the four ports and calibration names (plus the four
bimanual variants). This module owns that assembly so the two call sites
stay byte-for-byte identical; each caller then instantiates the concrete
robot/teleop devices from the returned configs.

The camera wiring is the only difference between the two callers: recording
puts the session's cameras on the (left) follower arm; teleoperation passes
no cameras at all. Callers that want cameras pass ``cameras=<dict>``; callers
that don't (teleop) leave it as ``None`` and the follower config is built
without a ``cameras`` kwarg — preserving the exact object each site built
before this module existed.

Inference (rollout.py) drives followers only and assembles its robot config
as subprocess CLI args, not as config objects, so it does not use this
module — the follower-only asymmetry lives there, not here.
"""

from pathlib import Path

from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig

from .config import (
    bimanual_base_id,
    setup_calibration_files,
    stage_bimanual_calibrations,
)


def build_single_configs(request, cameras=None):
    """Build (robot_config, teleop_config) for a single leader/follower pair.

    Stages the selected library calibrations into lerobot's expected
    locations (via ``setup_calibration_files``) and returns a follower
    ``SO101FollowerConfig`` and a leader ``SO101LeaderConfig``. When
    ``cameras`` is provided it is wired onto the follower; when ``None`` the
    follower config is built without a ``cameras`` kwarg (teleoperation).
    """
    leader_config_name, follower_config_name = setup_calibration_files(
        request.leader_config, request.follower_config
    )

    if cameras is None:
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
        )
    else:
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
            cameras=cameras,
        )

    teleop_config = SO101LeaderConfig(
        port=request.leader_port,
        id=leader_config_name,
    )

    return robot_config, teleop_config


def build_bimanual_configs(request, cameras=None):
    """Build (robot_config, teleop_config) for a bimanual BiSO pair.

    Stages the four arbitrarily-named library calibrations into the BiSO
    "<base>_left/right.json" convention (via ``stage_bimanual_calibrations``)
    and returns a ``BiSOFollowerConfig`` / ``BiSOLeaderConfig`` pair pointed
    at the per-device staging dirs. When ``cameras`` is provided it is wired
    onto the left follower arm; when ``None`` the left follower arm is built
    without a ``cameras`` kwarg (teleoperation).
    """
    base = bimanual_base_id(request.robot_name)
    leader_staging, follower_staging, _ = stage_bimanual_calibrations(
        base,
        request.leader_config,
        request.right_leader_config,
        request.follower_config,
        request.right_follower_config,
    )

    if cameras is None:
        left_follower = SO101FollowerConfig(port=request.follower_port)
    else:
        left_follower = SO101FollowerConfig(port=request.follower_port, cameras=cameras)

    robot_config = BiSOFollowerConfig(
        id=base,
        calibration_dir=Path(follower_staging),
        left_arm_config=left_follower,
        right_arm_config=SO101FollowerConfig(port=request.right_follower_port),
    )
    teleop_config = BiSOLeaderConfig(
        id=base,
        calibration_dir=Path(leader_staging),
        left_arm_config=SO101LeaderConfig(port=request.leader_port),
        right_arm_config=SO101LeaderConfig(port=request.right_leader_port),
    )

    return robot_config, teleop_config
