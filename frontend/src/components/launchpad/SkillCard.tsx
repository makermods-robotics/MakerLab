import React from "react";
import { ModelItem } from "@/lib/modelsApi";
import { policyTypeDisplayName } from "@/components/training/types";
import sockThumb from "@/assets/skill-sock-orange.jpg";
import bottleThumb from "@/assets/skill-bottle.jpg";
import towelThumb from "@/assets/skill-towel.jpg";
import stackCubesThumb from "@/assets/skill-stack-cubes.jpg";

/** Synthetic ids for skill previews that haven't been trained yet — no Hub
 * repo or job backs any of these, so none ever comes from `/models`. See WIP
 * handling in SkillSlider/SkillDetailDialog. */
export const WIP_SKILL_IDS = {
  bottleCap: "wip/bottle-cap-removal",
  towelFold: "wip/towel-fold",
  stackCubes: "wip/stack-cubes",
} as const;

const WIP_SKILL_ID_SET: ReadonlySet<string> = new Set(
  Object.values(WIP_SKILL_IDS),
);

/** True when `id` is one of the not-yet-trained WIP preview cards. */
export function isWipSkillId(id: string): boolean {
  return WIP_SKILL_ID_SET.has(id);
}

/** Curated preview media for featured skills, keyed by Hub repo id. */
const SKILL_THUMBNAILS: Record<string, string> = {
  "makermods/act_makermods_sock_2_only_more_orange_2026-07-16_22-14-55":
    sockThumb,
  [WIP_SKILL_IDS.bottleCap]: bottleThumb,
  [WIP_SKILL_IDS.towelFold]: towelThumb,
  [WIP_SKILL_IDS.stackCubes]: stackCubesThumb,
};

/** The curated preview image for a skill, or undefined when it has none. */
export function skillThumbnail(m: ModelItem): string | undefined {
  return SKILL_THUMBNAILS[m.hf_repo_id ?? m.id];
}

/** Curated human-readable names for featured skills, keyed by Hub repo id —
 * shown in place of the raw policy-run name. */
const SKILL_DISPLAY_NAMES: Record<string, string> = {
  "makermods/act_makermods_sock_2_only_more_orange_2026-07-16_22-14-55":
    "Sorting socks",
  [WIP_SKILL_IDS.bottleCap]: "Opening bottle caps",
  [WIP_SKILL_IDS.towelFold]: "Folding towels",
  [WIP_SKILL_IDS.stackCubes]: "Stacking cubes",
};

/** The marketplace provenance of a skill card. "wip" marks a preview card for
 * a skill that hasn't been trained yet — no repo/job backs it. */
export type SkillBadge = "mine" | "makermods" | "community" | "wip";

/** 16000 -> "16k", 950 -> "950" — matches the models card's compact form. */
export const formatCount = (n: number): string => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
};

/** The Hub namespace a skill lives under, or null for a bare local run id (no
 * "/"), which belongs to the logged-in user by construction. */
export function skillNamespace(m: ModelItem): string | null {
  const src = m.hf_repo_id ?? m.id;
  return src.includes("/") ? src.split("/")[0] : null;
}

/** MINE when the skill is in the user's namespace (or a bare local run they own);
 * MAKERMODS for the makermods org; COMMUNITY otherwise. Author == the namespace,
 * matched case-insensitively (mirrors DatasetInfoCard's useCanEditHub). */
export function classifySkill(
  m: ModelItem,
  username?: string | null,
): SkillBadge {
  const ns = skillNamespace(m);
  if (ns === null) return "mine";
  if (username && ns.toLowerCase() === username.toLowerCase()) return "mine";
  if (ns.toLowerCase() === "makermods") return "makermods";
  return "community";
}

/** True when the skill belongs in the user's library: any local checkpoint
 * (local/both) or a Hub repo in their own namespace. */
export function isMineSkill(m: ModelItem, username?: string | null): boolean {
  if (m.source === "local" || m.source === "both") return true;
  return classifySkill(m, username) === "mine";
}

/** The card/dialog title: a curated display name when the skill has one,
 * otherwise the name segment only (Hub rows carry the full "namespace/name" in
 * `name`), never an empty string. */
export function skillTitle(m: ModelItem): string {
  const curated = SKILL_DISPLAY_NAMES[m.hf_repo_id ?? m.id];
  if (curated) return curated;
  const raw = m.name || m.id;
  return raw.includes("/") ? (raw.split("/").pop() ?? raw) : raw;
}

const BADGE_LABEL: Record<SkillBadge, string> = {
  mine: "MINE",
  makermods: "MAKERMODS SUPPORTED",
  community: "COMMUNITY",
  wip: "WIP",
};

const BADGE_CLASS: Record<SkillBadge, string> = {
  mine: "border-transparent bg-primary text-primary-foreground",
  makermods: "border-ring bg-transparent text-foreground",
  community: "border-border bg-transparent text-muted-foreground",
  wip: "border-warn/40 bg-transparent text-warn",
};

/** Provenance pill (MINE / MAKERMODS / COMMUNITY) — token-styled, works in both
 * themes. */
export const SkillBadgePill: React.FC<{ badge: SkillBadge }> = ({ badge }) => (
  <span
    className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.06em] ${BADGE_CLASS[badge]}`}
  >
    {BADGE_LABEL[badge]}
  </span>
);

/** A small mono stat chip (policy type · steps · private). */
const Stat: React.FC<{ children: React.ReactNode; tone?: "amber" }> = ({
  children,
  tone,
}) => (
  <span
    className={`rounded border border-border px-1.5 py-0.5 font-mono text-[10.5px] ${
      tone === "amber" ? "text-warn" : "text-muted-foreground"
    }`}
  >
    {children}
  </span>
);

export interface SkillCardProps {
  model: ModelItem;
  badge: SkillBadge;
  onOpen: (model: ModelItem) => void;
}

/**
 * One skill in the launchpad slider. Preview media slot + title + author +
 * whatever real stats the /models payload carries (policy type, step count,
 * private flag) — no fabricated likes/downloads (the API exposes neither).
 * Click opens the skill detail dialog.
 */
const SkillCard: React.FC<SkillCardProps> = ({ model, badge, onOpen }) => {
  const ns = skillNamespace(model);
  const policy = model.policy_type
    ? policyTypeDisplayName(model.policy_type)
    : null;
  const thumbnail = skillThumbnail(model);

  return (
    <button
      type="button"
      onClick={() => onOpen(model)}
      className="group flex w-64 shrink-0 snap-start flex-col overflow-hidden rounded-lg border border-border bg-card text-left shadow-1 transition-colors hover:border-ring focus-visible:border-ring focus-visible:outline-none"
      aria-label={`Open skill ${skillTitle(model)}`}
    >
      {thumbnail ? (
        <img
          src={thumbnail}
          alt={`${skillTitle(model)} rollout preview`}
          className="aspect-[4/3] w-full object-cover"
        />
      ) : (
        <div
          className="media-slot aspect-[4/3] w-full"
          data-label="rollout preview"
        />
      )}
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex flex-col items-start gap-1.5">
          <span className="w-full min-w-0 truncate font-display font-semibold tracking-tight">
            {skillTitle(model)}
          </span>
          <SkillBadgePill badge={badge} />
        </div>
        <span className="truncate font-mono text-[11px] text-muted-foreground">
          {ns ?? "local checkpoint"}
        </span>
        <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
          {policy && <Stat>{policy}</Stat>}
          {model.steps != null && <Stat>{formatCount(model.steps)} steps</Stat>}
          {model.private && <Stat tone="amber">private</Stat>}
        </div>
      </div>
    </button>
  );
};

export default SkillCard;
