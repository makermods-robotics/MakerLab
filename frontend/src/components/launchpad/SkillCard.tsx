import React from "react";
import { ModelItem } from "@/lib/modelsApi";
import { policyTypeDisplayName } from "@/components/training/types";

/** The marketplace provenance of a skill card. */
export type SkillBadge = "mine" | "makermods" | "community";

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

/** The card/dialog title: the name segment only (Hub rows carry the full
 * "namespace/name" in `name`), never an empty string. */
export function skillTitle(m: ModelItem): string {
  const raw = m.name || m.id;
  return raw.includes("/") ? (raw.split("/").pop() ?? raw) : raw;
}

const BADGE_LABEL: Record<SkillBadge, string> = {
  mine: "MINE",
  makermods: "MAKERMODS",
  community: "COMMUNITY",
};

const BADGE_CLASS: Record<SkillBadge, string> = {
  mine: "border-transparent bg-primary text-primary-foreground",
  makermods: "border-ring bg-transparent text-foreground",
  community: "border-border bg-transparent text-muted-foreground",
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

  return (
    <button
      type="button"
      onClick={() => onOpen(model)}
      className="group flex w-64 shrink-0 snap-start flex-col overflow-hidden rounded-lg border border-border bg-card text-left shadow-1 transition-colors hover:border-ring focus-visible:border-ring focus-visible:outline-none"
      aria-label={`Open skill ${skillTitle(model)}`}
    >
      <div
        className="media-slot aspect-[4/3] w-full"
        data-label="rollout preview"
      />
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <span className="min-w-0 truncate font-display font-semibold tracking-tight">
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
