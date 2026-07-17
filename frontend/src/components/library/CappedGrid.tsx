import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { LIBRARY_GRID } from "./LibraryToolbar";

/** One row of the shared 3-up library grid. */
const CAP = 3;

/** Total height of the reserved grid block: the 16.5rem row + the 1.875rem
 * footer slot ("Show all" / Untracked / spacer) and its gutter. Empty/no-match
 * states adopt it so a library never shrinks below a populated one. Keep in
 * sync with the literal classes below — Tailwind only generates classes it
 * can see as literals. */
export const GRID_MIN_H = "min-h-[18.875rem]";

/**
 * The shared library grid, capped at one row. Every studio library (datasets,
 * training jobs, models) renders at most three cards by default; anything past
 * that stays hidden behind a "Show all" toggle. The row is always reserved
 * (fixed row height, blank cells when there aren't enough cards) so the three
 * panels' libraries keep one uniform height however large a collection grows.
 */
const CappedGrid: React.FC<{
  /** Pre-keyed cards, already sorted newest-first by the caller. */
  items: React.ReactNode[];
  /** Reserve the fixed row even when items don't fill it (the main
   * libraries). False for nested grids (e.g. Untracked) that should hug
   * their content. */
  reserveRows?: boolean;
  /** Hold the 30px footer slot with an invisible spacer when there's no
   * "Show all" — keeps every library's total height identical. False when
   * the caller renders its own footer row in that slot (jobs' Untracked). */
  footerSpacer?: boolean;
}> = ({ items, reserveRows = true, footerSpacer = true }) => {
  const [expanded, setExpanded] = useState(false);
  const overflow = items.length - CAP;
  const shown = expanded || overflow <= 0 ? items : items.slice(0, CAP);
  return (
    <div className="space-y-2">
      <div
        className={cn(
          LIBRARY_GRID,
          "auto-rows-[16.5rem]",
          reserveRows && "grid-rows-[16.5rem]",
        )}
      >
        {shown}
      </div>
      {overflow > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex h-[1.875rem] w-full items-center justify-center gap-1 rounded-md border border-dashed border-border text-xs font-medium text-muted-foreground transition-colors hover:border-muted-foreground/40 hover:text-foreground"
        >
          <ChevronDown
            className={cn(
              "h-3.5 w-3.5 transition-transform",
              expanded && "rotate-180",
            )}
          />
          {expanded ? "Show less" : `Show all ${items.length}`}
        </button>
      ) : reserveRows && footerSpacer ? (
        <div aria-hidden className="h-[1.875rem]" />
      ) : null}
    </div>
  );
};

export default CappedGrid;
