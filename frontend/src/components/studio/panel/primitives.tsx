import React from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/** Slide animation shared by every studio collapsible — the panels' entry
 * forms and all three libraries. The single home for this animation string;
 * don't copy it into components. */
export const SLIDE =
  "overflow-hidden data-[state=open]:animate-collapsible-down data-[state=closed]:animate-collapsible-up";

/** Numbered studio panel header: mono step digit + title, identical across
 * Collect, Train, and Deploy. */
export const PanelHeader: React.FC<{
  step: string;
  title: string;
  /** Optional trailing element (e.g. a resolving spinner). */
  children?: React.ReactNode;
}> = ({ step, title, children }) => (
  <div className="flex items-baseline gap-2">
    <span className="font-mono text-xs text-muted-foreground">{step}</span>
    <h2 className="text-base">{title}</h2>
    {children}
  </div>
);

/**
 * The panels' shared entry control — a quiet select-style row (same height,
 * border, and radius as a shadcn SelectTrigger) so Collect's "Record new
 * dataset", Train's "Start a new training", and Deploy's real skill <Select>
 * read as the same control. Used as a CollapsibleTrigger (asChild) that
 * slides the panel's form open in place.
 */
export const PanelEntryControl = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    open: boolean;
    /** Tiny status-dot accent (e.g. bg-red-500 for record). */
    dotClassName?: string;
  }
>(({ open, dotClassName, children, className, ...props }, ref) => (
  <button
    ref={ref}
    type="button"
    className={cn(
      "flex h-10 w-full items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background transition-colors hover:border-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
      className,
    )}
    {...props}
  >
    {dotClassName ? (
      <span
        aria-hidden
        className={cn("h-2 w-2 shrink-0 rounded-full", dotClassName)}
      />
    ) : null}
    <span className="truncate">{children}</span>
    <ChevronDown
      className={cn(
        "ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform",
        open && "rotate-180",
      )}
    />
  </button>
));
PanelEntryControl.displayName = "PanelEntryControl";

/** Pins a panel's library to the bottom of its column behind a hairline, so
 * the three libraries sit at the same level across the studio grid. */
export const LibrarySection: React.FC<{
  className?: string;
  children: React.ReactNode;
}> = ({ className, children }) => (
  <div className={cn("mt-auto border-t border-border pt-5", className)}>
    {children}
  </div>
);
