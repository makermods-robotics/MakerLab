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
"""
Bus class for vendored auto-calibration.

The `lerobot-MakerMods` fork builds the auto-calibration methods into its
`FeetechMotorsBus` via a mixin. MakerLab tracks upstream lerobot, so instead of
patching lerobot we compose the vendored mixin with MakerLab's *upstream*
`FeetechMotorsBus` here. The mixin only relies on the upstream bus's public API
(`read`/`write`/`sync_write`/`enable_torque`/`disable_torque`/`motors`/
`model_resolution_table`/`port_handler`/`packet_handler`), so this is additive —
nothing in lerobot is modified.
"""

from lerobot.motors.feetech import FeetechMotorsBus

from .auto_calibration import COMM_ERR, FeetechCalibrationMixin

__all__ = ["AutoCalBus", "COMM_ERR"]


class AutoCalBus(FeetechCalibrationMixin, FeetechMotorsBus):
    """Upstream FeetechMotorsBus extended with the auto-calibration mixin."""
