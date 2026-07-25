import React from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

/**
 * Fold header for one section inside a library ("This machine" / "On Hub, not
 * imported"): the section label, its count, the fold chevron, and — so a live
 * run is never lost behind a closed section — an optional running tally that
 * stays visible while the section is collapsed. Mirrors LibraryHeader one
 * level down: same label→count→chevron order, at the subsection's smaller
 * type. Must render inside its section's <Collapsible>.
 */
const LibrarySectionHeader: React.FC<{
  title: string;
  count: number;
  open: boolean;
  /** Live items in this section. Omit where the section has no running
   * concept — the model library's sections are all finished artifacts. */
  running?: number;
}> = ({ title, count, open, running = 0 }) => (
  <CollapsibleTrigger className="flex w-full items-center gap-1.5 px-2 py-0.5 text-muted-foreground transition-colors hover:text-foreground">
    <h4 className="text-[10px] font-medium uppercase tracking-wide">{title}</h4>
    <span className="rounded bg-muted px-1 py-px font-mono text-[10px]">
      {count}
    </span>
    <ChevronDown
      className={cn("h-3 w-3 transition-transform", open && "rotate-180")}
    />
    {running > 0 ? (
      <span className="flex items-center gap-1 text-[10px] font-medium text-ok">
        <Loader2 className="h-3 w-3 animate-spin" />
        {running} running
      </span>
    ) : null}
  </CollapsibleTrigger>
);

export default LibrarySectionHeader;
