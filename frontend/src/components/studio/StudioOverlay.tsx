import React, { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import BrandMark from "@/components/BrandMark";
import RobotCorner from "@/components/launchpad/RobotCorner";
import HfAuthChip from "@/components/landing/HfAuthChip";
import CollectPanel from "@/components/studio/CollectPanel";
import TrainPanel from "@/components/studio/TrainPanel";
import { useStudio } from "@/contexts/StudioContext";
import { cn } from "@/lib/utils";

/**
 * The fullscreen skill studio — slides up over the Launchpad when the
 * "+ New Skill" banner (or any Fine-tune action) opens it. Two panels:
 * Collect then Train. Running a skill is NOT a panel — every "run this model"
 * entry point calls InferenceLaunchProvider, whose modal is hosted above the
 * router, so Run works the same inside the studio and out on the Launchpad.
 * Stays mounted so panel state survives close/reopen within a visit.
 */
const StudioOverlay: React.FC = () => {
  const { open, activePanel, closeStudio } = useStudio();
  const overlayRef = useRef<HTMLDivElement>(null);

  // Lock page scroll while the studio is up.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  // ESC closes the studio — unless a dialog above it already handled the key
  // (Radix prevents default when it closes its own layer), or the key was
  // pressed inside a portaled dialog that stays open (the recording session
  // dialog focus-traps and repurposes ESC as "finish & keep" — its keypress
  // must never fall through and close the studio underneath).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      const target = e.target as HTMLElement | null;
      const dialogAbove = target?.closest(
        '[role="dialog"], [role="alertdialog"]',
      );
      if (dialogAbove && dialogAbove !== overlayRef.current) return;
      closeStudio();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, closeStudio]);

  return (
    <div
      ref={overlayRef}
      role="dialog"
      aria-label="Skill studio"
      aria-hidden={!open}
      className={cn(
        "fixed inset-0 z-40 flex flex-col bg-background transition-transform duration-500 ease-std motion-reduce:transition-none",
        open ? "translate-y-0" : "pointer-events-none translate-y-full",
      )}
    >
      <header className="flex items-center gap-3 border-b border-border px-4 py-3 sm:px-6">
        <BrandMark size="md" />
        <span className="hidden rounded border border-border px-1.5 py-0.5 font-orbitron text-[11px] font-black uppercase tracking-[0.08em] text-muted-foreground sm:inline">
          by MakerMods
        </span>
        <span className="eyebrow ml-2 hidden md:inline">Skill studio</span>
        <span className="flex-1" />
        <span className="hidden sm:inline-flex">
          <HfAuthChip />
        </span>
        <RobotCorner />
        <Button
          variant="ghost"
          size="sm"
          aria-label="Close studio"
          onClick={closeStudio}
          className="h-8 w-8 p-0"
        >
          <X className="h-4 w-4" />
        </Button>
      </header>

      {/* Train's model library reads JobsDataProvider, mounted a level up on
          the Launchpad — the library drawer's Runs tab needs it too. */}
      <div className="grid flex-1 grid-cols-1 gap-px overflow-y-auto bg-border lg:grid-cols-2 lg:overflow-hidden">
        <section
          aria-label="Collect dataset"
          className={cn(
            "flex min-h-0 flex-col bg-background lg:overflow-y-auto",
            activePanel === "collect" && "ring-1 ring-inset ring-ring/20",
          )}
        >
          <CollectPanel />
        </section>
        <section
          aria-label="Train"
          className={cn(
            "flex min-h-0 flex-col bg-background lg:overflow-y-auto",
            activePanel === "train" && "ring-1 ring-inset ring-ring/20",
          )}
        >
          <TrainPanel />
        </section>
      </div>
    </div>
  );
};

export default StudioOverlay;
