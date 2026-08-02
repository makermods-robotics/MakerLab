/**
 * The CLIENT-side leader configuration, for robots whose `leader_source` is
 * `"remote"`.
 *
 * When the leader arm is plugged into this laptop while the follower + cameras
 * live on a remote host, the leader's port and calibration are NOT part of the
 * remote's robot record — that record describes the remote's own hardware
 * (docs/remote-portal/SPEC.md, "Robot record schema change"). They belong to
 * this machine and are handed to the leader-bridge in the
 * `POST /leader-bridge/start` payload instead.
 *
 * So they live here: browser-local, keyed by the remote host + robot name, so
 * one laptop can drive several servers without their leader setups colliding.
 *
 * Field names deliberately mirror the `/leader-bridge/start` body so a caller
 * can spread this straight into the request.
 */

export interface LocalLeaderConfig {
  leader_port: string;
  leader_config: string;
  /** Bimanual only — the right-hand leader pair. */
  right_leader_port: string;
  right_leader_config: string;
}

export const EMPTY_LOCAL_LEADER: LocalLeaderConfig = {
  leader_port: "",
  leader_config: "",
  right_leader_port: "",
  right_leader_config: "",
};

const KEY_PREFIX = "makerlab.localLeader";

const keyFor = (host: string, robotName: string): string =>
  `${KEY_PREFIX}::${host}::${robotName}`;

export const readLocalLeader = (
  host: string,
  robotName: string,
): LocalLeaderConfig => {
  if (typeof window === "undefined" || !robotName) return EMPTY_LOCAL_LEADER;
  try {
    const raw = window.localStorage.getItem(keyFor(host, robotName));
    if (!raw) return EMPTY_LOCAL_LEADER;
    const parsed = JSON.parse(raw);
    return {
      leader_port: String(parsed.leader_port ?? ""),
      leader_config: String(parsed.leader_config ?? ""),
      right_leader_port: String(parsed.right_leader_port ?? ""),
      right_leader_config: String(parsed.right_leader_config ?? ""),
    };
  } catch {
    return EMPTY_LOCAL_LEADER;
  }
};

export const writeLocalLeader = (
  host: string,
  robotName: string,
  value: LocalLeaderConfig,
) => {
  if (typeof window === "undefined" || !robotName) return;
  try {
    window.localStorage.setItem(keyFor(host, robotName), JSON.stringify(value));
  } catch {
    // Storage unavailable (private mode, quota) — non-fatal.
  }
};
