import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { LIBRARY_GRID } from "./LibraryToolbar";

/** One row of the shared 3-up library grid — how many cards show before the
 * "Show all" toggle. Exported so a caller can tell whether its grid overflows,
 * and thus whether to fold its own footer toggle under "Show all" instead of
 * stacking a second footer button. */
export const LIBRARY_ROW_CAP = 3;

/** Total height of the reserved grid block: the 16.5rem row + the 1.875rem
 * footer slot ("Show all" / Untracked / spacer) and its gutter. Empty/no-match
 * states adopt it so a library never shrinks below a populated one. Keep in
 * sync with the literal classes below — Tailwind only generates classes it
 * can see as literals. */
export const GRID_MIN_H = "min-h-[18.875rem]";

/**
 * The shared library grid, capped at one row. Every studio library (datasets,
 * training jobs, models) renders one row of three cards by default; anything
 * past that stays hidden behind a "Show all" toggle. By default the row is
 * reserved at a fixed height (blank cells when there aren't enough cards) so
 * datasets and jobs keep one uniform height however large a collection grows.
 * Models opts into `flexHeight` instead, letting its taller cards size to
 * content.
 */
const CappedGrid: React.FC<{
  /** Pre-keyed cards, already sorted newest-first by the caller. */
  items: React.ReactNode[];
  /** Reserve the fixed row even when items don't fill it (the main
   * libraries). False for nested grids (e.g. Untracked) that should hug
   * their content. */
  reserveRows?: boolean;
  /** Size rows to their content instead of the fixed 16.5rem block. Models
   * opts in because its cards render every affordance and must not clip;
   * datasets/jobs stay fixed (the default) so their libraries keep a uniform
   * height. In flexible mode the fixed-height reserve is skipped — each row
   * grows to its tallest card (items-stretch still equalizes cards within a
   * row); the footer-slot reserve (spacer/Show-all) is unaffected. */
  flexHeight?: boolean;
  /** Hold the 30px footer slot with an invisible spacer when there's no
   * "Show all" — keeps every library's total height identical. False when
   * the caller renders its own footer row in that slot (jobs' Untracked). */
  footerSpacer?: boolean;
  /** Optionally control the expanded state from the parent (jobs lifts it so
   * its Untracked toggle can appear only once "Show all" is open). Omit for
   * the self-managed default used by datasets and models. */
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}> = ({
  items,
  reserveRows = true,
  footerSpacer = true,
  flexHeight = false,
  expanded: expandedProp,
  onExpandedChange,
}) => {
  const [expandedState, setExpandedState] = useState(false);
  const expanded = expandedProp ?? expandedState;
  const setExpanded = (value: boolean) => {
    onExpandedChange?.(value);
    if (expandedProp === undefined) setExpandedState(value);
  };
  const overflow = items.length - LIBRARY_ROW_CAP;
  const shown =
    expanded || overflow <= 0 ? items : items.slice(0, LIBRARY_ROW_CAP);
  return (
    <div className="space-y-2">
      <div
        className={LIBRARY_GRID}
        // Row sizing is a runtime choice, so drive it inline rather than via a
        // Tailwind arbitrary class. Fixed mode (default) reproduces the former
        // auto-rows-[16.5rem] / grid-rows-[16.5rem] classes exactly, keeping
        // the uniform reserved block datasets/jobs share. Flexible mode sizes
        // rows to content (auto) and skips the fixed reserve so tall cards
        // never clip — items-stretch still equalizes cards within a row.
        style={
          flexHeight
            ? { gridAutoRows: "auto" }
            : {
                gridAutoRows: "16.5rem",
                ...(reserveRows ? { gridTemplateRows: "16.5rem" } : {}),
              }
        }
      >
        {shown}
      </div>
      {overflow > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
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
