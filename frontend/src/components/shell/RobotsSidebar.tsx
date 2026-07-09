import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Settings, Plus } from "lucide-react";
import { useRobots, RobotMode } from "@/hooks/useRobots";
import CreateRobotDialog from "@/components/landing/CreateRobotDialog";
import HfAuthChip from "@/components/landing/HfAuthChip";
import { ThemeToggle } from "@/components/shell/ThemeToggle";
import Logo from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Robot status line: mode + readiness derived from the record. */
const statusLine = (mode: RobotMode, isClean: boolean | undefined) =>
  `${mode} · ${isClean ? "ready" : "needs calibration"}`;

/**
 * The persistent robots rail on stage pages (Collect / Train & Deploy / Market).
 * Selection flows through the useRobots singleton store so every consumer
 * (pages, dialogs) stays in sync; the gear deep-links into the robot's full
 * settings surface (ports / calibration / cameras / motor power).
 */
const RobotsSidebar: React.FC = () => {
  const navigate = useNavigate();
  const {
    records,
    selectedName,
    availableNames,
    selectRobot,
    createRobot,
  } = useRobots();
  const [createOpen, setCreateOpen] = useState(false);
  // The store keeps records as a name-keyed map; render in stable name order.
  const robots = Object.values(records).sort((a, b) =>
    a.name.localeCompare(b.name)
  );

  return (
    <aside className="fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-border bg-background">
      <div className="px-4 pb-4 pt-5">
        <Logo className="px-2" />
      </div>
      <div className="eyebrow px-6 pb-2">Robots</div>
      <nav className="flex-1 overflow-y-auto px-3" aria-label="Robots">
        {robots.length === 0 && (
          <p className="px-3 py-2 text-sm text-muted-foreground">
            No robots yet — create one to start.
          </p>
        )}
        {robots.map((r) => {
          const active = r.name === selectedName;
          return (
            <div
              key={r.name}
              className={cn(
                "group mb-0.5 grid w-full grid-cols-[10px_1fr_auto] items-center gap-2.5 rounded-md px-2.5 py-2 text-left",
                active
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-accent"
              )}
            >
              <span
                className={cn(
                  "h-[7px] w-[7px] rounded-full",
                  r.is_clean ? "bg-ok" : "bg-warn"
                )}
              />
              <button
                type="button"
                className="min-w-0 text-left"
                onClick={() => selectRobot(r.name)}
              >
                <span className="block truncate text-[13px] font-medium">
                  {r.name}
                </span>
                <span
                  className={cn(
                    "block truncate text-[11px]",
                    active
                      ? "text-primary-foreground/70"
                      : "text-muted-foreground"
                  )}
                >
                  {statusLine(r.mode, r.is_clean)}
                </span>
              </button>
              <button
                type="button"
                aria-label={`Configure ${r.name}`}
                className={cn(
                  "grid h-7 w-7 place-items-center rounded-md opacity-60 hover:opacity-100",
                  active ? "hover:bg-primary-foreground/15" : "hover:bg-accent"
                )}
                onClick={() =>
                  navigate("/calibration", { state: { robot_name: r.name } })
                }
              >
                <Settings className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
        <Button
          variant="outline"
          size="sm"
          className="mt-2 w-full"
          onClick={() => setCreateOpen(true)}
        >
          <Plus className="h-3.5 w-3.5" /> New robot
        </Button>
      </nav>
      <div className="flex items-center justify-between gap-2 border-t border-border p-3">
        <HfAuthChip />
        <ThemeToggle />
      </div>
      <CreateRobotDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        availableNames={availableNames}
        defaultMode="single"
        onCreateNew={async (name, mode) => {
          const ok = await createRobot(name, mode);
          if (ok) selectRobot(name);
          return ok;
        }}
      />
    </aside>
  );
};

export default RobotsSidebar;
