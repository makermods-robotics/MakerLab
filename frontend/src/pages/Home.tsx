import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Download, Grid2X2, Plus } from "lucide-react";
import BoothHero from "@/components/home/BoothHero";
import CreateRobotDialog from "@/components/landing/CreateRobotDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useRobots, type RobotRecord } from "@/hooks/useRobots";
import { cn } from "@/lib/utils";

const EXIT_MS = 600;

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const cameraMeta = (count: number) => `${count} ${count === 1 ? "cam" : "cams"}`;

const robotMeta = (robot: RobotRecord) => {
  if (!robot.is_clean) return "needs calibration";
  const leader = robot.leader_port?.trim() || "leader unset";
  return `${leader} · ${cameraMeta(robot.cameras?.length ?? 0)}`;
};

interface ActionCardProps {
  icon: React.ReactNode;
  label: string;
  to?: string;
  onClick?: () => void;
}

const actionClass =
  "group flex min-h-[78px] flex-col justify-between rounded-lg border border-border bg-card px-3.5 py-3 text-left text-foreground shadow-1 transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2";

const ActionCard: React.FC<ActionCardProps> = ({ icon, label, to, onClick }) => {
  const content = (
    <>
      <span className="text-muted-foreground transition-colors group-hover:text-foreground">
        {icon}
      </span>
      <span className="text-[13.5px] font-medium">{label}</span>
    </>
  );

  if (to) {
    return (
      <Link to={to} className={actionClass}>
        {content}
      </Link>
    );
  }

  return (
    <button type="button" onClick={onClick} className={actionClass}>
      {content}
    </button>
  );
};

interface RobotRowProps {
  robot: RobotRecord;
  onSelect: (robot: RobotRecord) => void;
}

const RobotRow: React.FC<RobotRowProps> = ({ robot, onSelect }) => (
  <button
    type="button"
    className="grid w-full grid-cols-[12px_1fr_auto] items-center gap-2.5 rounded-md px-3 py-2.5 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
    onClick={() => onSelect(robot)}
  >
    <span
      className={cn(
        "h-2 w-2 rounded-full",
        robot.is_clean ? "bg-ok" : "bg-warn"
      )}
      aria-hidden="true"
    />
    <span className="flex min-w-0 items-center gap-2">
      <span className="truncate text-[13.5px] font-medium">{robot.name}</span>
      <Badge variant="secondary" className="shrink-0 px-2 py-0 text-[10px]">
        {robot.mode}
      </Badge>
    </span>
    <span className="hidden max-w-[190px] truncate font-mono text-[11px] text-muted-foreground sm:block">
      {robotMeta(robot)}
    </span>
  </button>
);

const Home: React.FC = () => {
  const navigate = useNavigate();
  const {
    records,
    selectedName,
    availableNames,
    selectRobot,
    createRobot,
    isLoading,
  } = useRobots();
  const [createOpen, setCreateOpen] = useState(false);
  const [exitingName, setExitingName] = useState<string | null>(null);

  const robots = useMemo(
    () =>
      availableNames
        .map((name) => records[name])
        .filter((robot): robot is RobotRecord => Boolean(robot)),
    [availableNames, records]
  );

  const settingsState = useMemo(
    () => (selectedName ? { robot_name: selectedName } : undefined),
    [selectedName]
  );
  const isExiting = exitingName !== null;

  const openRobot = useCallback(
    (robot: RobotRecord) => {
      selectRobot(robot.name);
      if (prefersReducedMotion()) {
        navigate("/collect");
        return;
      }
      setExitingName(robot.name);
      window.setTimeout(() => navigate("/collect"), EXIT_MS);
    },
    [navigate, selectRobot]
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isExiting) return;
      const target = event.target as HTMLElement | null;
      const isTyping =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target?.isContentEditable;
      if (isTyping) return;

      if (event.key === "Enter" && robots.length > 0) {
        event.preventDefault();
        // Prefer the robot the user last worked with; fall back to the first.
        const preferred =
          robots.find((r) => r.name === selectedName) ?? robots[0];
        openRobot(preferred);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        setCreateOpen(true);
      }
      if ((event.metaKey || event.ctrlKey) && event.key === ",") {
        event.preventDefault();
        navigate("/calibration", { state: settingsState });
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isExiting, navigate, openRobot, robots, selectedName, settingsState]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <section
        data-exiting={isExiting}
        className={cn(
          "fixed inset-y-0 left-1/2 z-20 w-[min(700px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden border-r border-transparent bg-background transition-[left,width,transform,border-color] duration-[600ms] ease-std",
          "data-[exiting=true]:left-0 data-[exiting=true]:w-72 data-[exiting=true]:translate-x-0 data-[exiting=true]:border-border"
        )}
      >
        <div
          data-exiting={isExiting}
          className="flex min-h-screen flex-col items-center justify-center px-2.5 py-7 transition-opacity duration-300 data-[exiting=true]:pointer-events-none data-[exiting=true]:opacity-0"
        >
          <div className="w-full max-w-[520px] overflow-hidden transition-[max-height,opacity] duration-[600ms] ease-std data-[exiting=true]:max-h-0 data-[exiting=true]:opacity-0">
            <BoothHero className="block h-auto w-full" />
          </div>

          <header className="mt-3.5 flex flex-col items-center gap-1.5">
            <div className="flex items-center gap-2.5">
              <img
                src="/makermods/logo-mark.png"
                alt=""
                className="h-[22px] w-auto dark:hidden"
              />
              <img
                src="/makermods/logo-mark-white.png"
                alt=""
                className="hidden h-[22px] w-auto dark:block"
              />
              <h1 className="text-[19px] font-bold leading-none tracking-tight">
                MakerLab
              </h1>
            </div>
            <p className="text-[13px] text-muted-foreground">
              SO-101 workbench{" "}
              <span className="mx-1.5 text-muted-foreground/60">·</span>
              <Link
                to="/calibration"
                state={settingsState}
                className="underline-offset-4 hover:text-foreground hover:underline"
              >
                Settings
              </Link>
            </p>
          </header>

          <nav
            aria-label="Start"
            className="mt-6 grid w-full max-w-[520px] grid-cols-1 gap-2.5 px-2.5 sm:grid-cols-3 sm:px-0"
          >
            <ActionCard
              icon={<Plus className="h-4 w-4" />}
              label="New robot"
              onClick={() => setCreateOpen(true)}
            />
            <ActionCard
              icon={<Download className="h-4 w-4" />}
              label="Import from Hub"
              to="/market"
            />
            <ActionCard
              icon={<Grid2X2 className="h-4 w-4" />}
              label="Browse Market"
              to="/market"
            />
          </nav>

          <section className="mt-7 w-full max-w-[520px]" aria-label="Recent robots">
            <div className="flex items-baseline justify-between px-3 pb-2">
              <h2 className="text-[13px] font-semibold text-muted-foreground">
                Recent robots
              </h2>
              <span className="text-[12.5px] text-muted-foreground">
                {isLoading ? "Refreshing" : `View all (${robots.length})`}
              </span>
            </div>
            {robots.length > 0 ? (
              <div>
                {robots.map((robot) => (
                  <RobotRow key={robot.name} robot={robot} onSelect={openRobot} />
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-border bg-card px-4 py-5 text-center shadow-1">
                <p className="text-sm font-medium">No robots yet</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Create a robot profile to start collecting demonstrations.
                </p>
                <Button
                  className="mt-4"
                  variant="brand"
                  onClick={() => setCreateOpen(true)}
                >
                  <Plus className="h-4 w-4" /> New robot
                </Button>
              </div>
            )}
          </section>

          <p className="mt-8 text-center text-xs text-muted-foreground">
            <kbd className="rounded-[5px] border border-border bg-muted px-1.5 py-0.5 font-mono text-[11px] shadow-1">
              Enter
            </kbd>{" "}
            open robot
            <span className="mx-2 text-muted-foreground/60">·</span>
            <kbd className="rounded-[5px] border border-border bg-muted px-1.5 py-0.5 font-mono text-[11px] shadow-1">
              Cmd+N
            </kbd>{" "}
            new robot
            <span className="mx-2 text-muted-foreground/60">·</span>
            <kbd className="rounded-[5px] border border-border bg-muted px-1.5 py-0.5 font-mono text-[11px] shadow-1">
              Cmd+,
            </kbd>{" "}
            settings
          </p>
        </div>

        <div
          data-exiting={isExiting}
          aria-hidden={!isExiting}
          className="absolute inset-0 px-4 pb-4 pt-5 opacity-0 transition-opacity delay-200 duration-300 data-[exiting=true]:opacity-100"
        >
          <div className="flex items-center gap-2 px-2 pb-4">
            <img
              src="/makermods/logo-mark.png"
              alt=""
              className="h-[15px] w-auto dark:hidden"
            />
            <img
              src="/makermods/logo-mark-white.png"
              alt=""
              className="hidden h-[15px] w-auto dark:block"
            />
            <span className="text-[13.5px] font-semibold">MakerLab</span>
          </div>
          <div className="eyebrow px-2 pb-2">Robots</div>
          {robots.map((robot) => {
            const active = robot.name === exitingName;
            return (
              <div
                key={robot.name}
                className={cn(
                  "mb-0.5 grid grid-cols-[10px_1fr] items-center gap-2.5 rounded-md px-2.5 py-2",
                  active ? "bg-primary text-primary-foreground" : "text-foreground"
                )}
              >
                <span
                  className={cn(
                    "h-[7px] w-[7px] rounded-full",
                    robot.is_clean ? "bg-ok" : "bg-warn"
                  )}
                />
                <span className="min-w-0">
                  <span className="block truncate text-[13px] font-medium">
                    {robot.name}
                  </span>
                  <span
                    className={cn(
                      "block truncate text-[11px]",
                      active
                        ? "text-primary-foreground/70"
                        : "text-muted-foreground"
                    )}
                  >
                    {robot.mode} · {robot.is_clean ? "ready" : "needs calibration"}
                  </span>
                </span>
              </div>
            );
          })}
        </div>
      </section>

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
    </main>
  );
};

export default Home;
