// Per-motor target range (max - min, in raw motor steps) by device_type.
// Derived from observed SO-101 calibration files; values sit slightly below
// the smallest observed good range so a real calibration clears the 98% bar.
const SO101_LEADER_TARGETS: Record<string, number> = {
  shoulder_pan: 2400,
  shoulder_lift: 2300,
  elbow_flex: 2150,
  wrist_flex: 2250,
  wrist_roll: 3700,
  gripper: 1150,
};

const SO101_FOLLOWER_TARGETS: Record<string, number> = {
  shoulder_pan: 2400,
  shoulder_lift: 2300,
  elbow_flex: 2150,
  wrist_flex: 2250,
  wrist_roll: 3700,
  gripper: 1400,
};

const TARGETS_BY_DEVICE_TYPE: Record<string, Record<string, number>> = {
  teleop: SO101_LEADER_TARGETS,
  robot: SO101_FOLLOWER_TARGETS,
};

// Continuous full-turn joints. Official lerobot-calibrate excludes these from
// the range sweep and hardcodes 0-4095 (lerobot so_follower.py/so_leader.py
// calibrate()); the makerlab backend mirrors that (FULL_TURN_MOTORS in
// makerlab/calibrate.py). They must NOT be swept — rolling past the encoder wrap
// used to trip the discontinuity check — so their checkmark is always green.
const FULL_TURN_MOTORS = new Set(["wrist_roll"]);

const RANGE_TOLERANCE = 0.98;

export function isMotorRangeComplete(
  deviceType: string | null | undefined,
  motor: string,
  rangeAchieved: number
): boolean {
  if (FULL_TURN_MOTORS.has(motor)) return true;
  if (!deviceType) return false;
  const target = TARGETS_BY_DEVICE_TYPE[deviceType]?.[motor];
  if (!target) return false;
  return rangeAchieved >= target * RANGE_TOLERANCE;
}
