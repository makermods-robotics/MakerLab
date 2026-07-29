/** Human formatting for dataset metadata, shared by the dataset info card and
 * the Collect panel's library cards. */

/** 16723 -> "16.7k", 950 -> "950" */
export const formatCount = (n: number): string => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
};

/** frames ÷ fps, human-formatted: "~9 min", "~45 s", "~1 h 12 min" */
export const formatDuration = (
  frames: number,
  fps: number | null,
): string | null => {
  if (!fps || fps <= 0 || frames <= 0) return null;
  const seconds = frames / fps;
  if (seconds < 60) return `~${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `~${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m > 0 ? `~${h} h ${m} min` : `~${h} h`;
};

export const formatBytes = (bytes: number | null | undefined): string => {
  // Null-safe: an unknown size renders nothing rather than "null B". Callers
  // still gate the whole Size row on presence, so this is belt-and-suspenders.
  if (bytes == null) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

/** The per-arm prefix lerobot's bimanual wrappers add to each sub-arm feature. */
const ARM_PREFIX_RE = /^(left|right)_/;

/**
 * True when a dataset's `robot_type` names one of lerobot's bimanual wrapper
 * robots, whose observation keys carry the per-arm `left_`/`right_` prefix.
 *
 * lerobot's bimanual robots are all named `bi_*` (`bi_so_follower`,
 * `bi_openarm_follower`, `bi_rebot_b601_follower`) and every one of them
 * prefixes its sub-arm features, so the `bi_` convention is the signal. makerlab
 * itself only ever produces `bi_so_follower`; the broader test just means an
 * externally-recorded bimanual dataset reads correctly too.
 *
 * `robot_type` is optional in lerobot's metadata (`str | None`), so an absent
 * one is treated as NOT bimanual — a single-arm camera legitimately named
 * `left_side` must never be mangled, and leaving the real key on screen is the
 * safe failure.
 */
export const isBimanualRobotType = (
  robotType: string | null | undefined,
): boolean => robotType != null && robotType.startsWith("bi_");

/**
 * Camera names as they should be SHOWN to the user — display only.
 *
 * lerobot's BiSOFollower parks makerlab's cameras on the left arm and prefixes
 * each feature when it writes the dataset, so a camera the user named `front`
 * is stored as `observation.images.left_front`. That prefix is real, correct
 * data: a policy trained on those keys needs them verbatim at inference. This
 * helper ONLY shortens the rendered string — never a key that is compared,
 * sent, or persisted. Callers should keep the raw name as the row's tooltip.
 *
 * Single-arm datasets pass through untouched. In bimanual datasets only a
 * leading `left_`/`right_` is removed, and only the first one, so `left_left`
 * renders as `left` and a bare `left` stays `left`.
 *
 * Collisions keep the full name: if two cameras would strip to the same bare
 * name (`left_front` + `right_front`, or a bare `front` beside `left_front`),
 * every colliding entry renders in full rather than producing a duplicate list.
 * makerlab attaches cameras to the left arm only, so this is a defensive branch
 * for externally-recorded datasets. Mirrors `cameraMappings` in
 * components/landing/InferenceModal.tsx, which makes the same call for the
 * inference camera bindings.
 */
export const displayCameraNames = (
  cameras: string[],
  robotType: string | null | undefined,
): string[] => {
  if (!isBimanualRobotType(robotType)) return cameras;
  // Count what each name would strip to, so a collision can be detected before
  // committing to the shortened form.
  const bareCounts = new Map<string, number>();
  for (const cam of cameras) {
    const bare = cam.replace(ARM_PREFIX_RE, "");
    bareCounts.set(bare, (bareCounts.get(bare) ?? 0) + 1);
  }
  return cameras.map((cam) => {
    const bare = cam.replace(ARM_PREFIX_RE, "");
    return bareCounts.get(bare) === 1 ? bare : cam;
  });
};
