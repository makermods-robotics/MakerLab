import React, { useState } from "react";
import {
  Gamepad2,
  Plus,
  Settings,
  ChevronDown,
  Loader2,
  Pencil,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import CreateRobotDialog from "@/components/landing/CreateRobotDialog";
import TeleopDialog from "@/components/dialogs/TeleopDialog";
import RobotConfigDialog from "@/components/dialogs/RobotConfigDialog";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots, RobotRecord, RobotMode, robotSetupGap } from "@/hooks/useRobots";
import { cn } from "@/lib/utils";

/** Status dot: calibrated (ok) vs needs setup (warn ring). */
const StatusDot: React.FC<{ ready: boolean; className?: string }> = ({
  ready,
  className,
}) => (
  <span
    aria-hidden
    className={cn(
      "inline-block h-2 w-2 shrink-0 rounded-full",
      ready ? "bg-ok" : "border border-warn bg-transparent",
      className,
    )}
  />
);

/**
 * The robot corner — Layout D's always-visible robot control, one pill
 * cluster so the pieces read as a single unit: "+ Robot", an icon-only
 * Settings button, a chip with the active robot + dropdown (instant switch,
 * create, rename, delete), and a Teleop button as the rightmost segment.
 * Mounted on the Launchpad header AND inside the studio overlay header,
 * sharing state through useRobots' module-level store.
 */
const RobotCorner: React.FC<{ className?: string }> = ({ className }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const {
    records,
    selectedName,
    selectedRecord,
    availableNames,
    isLoading,
    refresh,
    selectRobot,
    createRobot,
    renameRobot,
    deleteRobot,
  } = useRobots();

  const [createOpen, setCreateOpen] = useState(false);
  // Robot settings window (ports, calibration, cameras, motor power).
  const [configOpen, setConfigOpen] = useState(false);
  const [configRobotName, setConfigRobotName] = useState<string | null>(null);
  const [teleopStarting, setTeleopStarting] = useState(false);
  const [teleopOpen, setTeleopOpen] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const openRename = () => {
    setRenameValue(selectedName ?? "");
    setRenameOpen(true);
  };

  // useRobots owns validation, API errors, and toasts for rename/delete —
  // these handlers only manage the dialogs (same split as CreateRobotDialog).
  const handleRenameConfirm = async () => {
    if (!selectedName) return;
    setRenaming(true);
    try {
      const ok = await renameRobot(selectedName, renameValue);
      if (ok) setRenameOpen(false);
    } finally {
      setRenaming(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!selectedName) return;
    await deleteRobot(selectedName);
    setDeleteOpen(false);
  };

  const hasRobots = availableNames.length > 0;

  // Open the Robot settings window for a robot. On close, re-fetch the shared
  // records — the window may have saved ports/cameras/torque or assigned
  // calibrations, and (unlike the old /calibration page) closing a dialog
  // doesn't remount anything that would refresh on its own.
  const openSettings = (name?: string | null) => {
    if (!name) return;
    setConfigRobotName(name);
    setConfigOpen(true);
  };

  const handleConfigOpenChange = (open: boolean) => {
    setConfigOpen(open);
    if (!open) refresh();
  };

  // Create → select (useRobots does this on success) → straight into the Robot
  // settings window so ports/calibration/cameras get configured (wireframe J1).
  const handleCreate = async (name: string, mode: RobotMode) => {
    const ok = await createRobot(name, mode);
    if (ok) {
      setCreateOpen(false);
      openSettings(name);
    }
    return ok;
  };

  // Ported verbatim from min_stable RobotConfigManager.handleTeleop — the
  // backend returns HTTP 200 with { success: false } for logical failures,
  // so gate on data.success, not just res.ok.
  const handleTeleop = async (robot: RobotRecord) => {
    setTeleopStarting(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/move-arm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          leader_port: robot.leader_port,
          follower_port: robot.follower_port,
          leader_config: robot.leader_config,
          follower_config: robot.follower_config,
          // Bimanual: include the mode + right arm so the backend builds a BiSO pair.
          mode: robot.mode,
          right_leader_port: robot.right_leader_port,
          right_follower_port: robot.right_follower_port,
          right_leader_config: robot.right_leader_config,
          right_follower_config: robot.right_follower_config,
          // Robot name → BiSO staging base id (bimanual). Names the per-session
          // staging dir; does not affect which calibration drives which arm.
          robot_name: robot.name,
        }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        // A success can carry a warn-but-allow arm-identity finding (e.g. the
        // arm's servos hold a different saved calibration). Make it visible.
        if (data.warning) {
          toast({
            title: "Started with a warning",
            description: data.warning,
            duration: 10000,
          });
        } else {
          toast({
            title: "Teleoperation started",
            description:
              data.message || `Started teleoperation for ${robot.name}.`,
          });
        }
        setTeleopOpen(true);
      } else {
        toast({
          title: "Couldn't start teleoperation",
          description: data.message || "Failed to start.",
          variant: "destructive",
        });
      }
    } catch {
      toast({
        title: "Connection error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    } finally {
      setTeleopStarting(false);
    }
  };

  const teleopDisabledReason = !selectedRecord
    ? "Select a robot first"
    : !selectedRecord.is_clean
      ? `${selectedRecord.name} ${robotSetupGap(selectedRecord)} — open Robot settings`
      : null;

  return (
    <div
      className={cn(
        "flex items-center gap-0.5 rounded-full border border-border bg-card p-0.5",
        className,
      )}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          {/* First run (no robots yet): the very first action in the app lives
              in this cluster, and studio copy points here — render it filled
              primary so "add a robot in the top-right corner" is findable at a
              glance instead of a ghost button to hunt for. */}
          <Button
            variant={hasRobots ? "ghost" : "default"}
            size="sm"
            onClick={() => setCreateOpen(true)}
            className="h-7 gap-1.5 rounded-full px-2.5"
          >
            <Plus className="h-3.5 w-3.5" />
            Robot
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Create robot</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <span>
            <Button
              variant="ghost"
              size="sm"
              disabled={!selectedName}
              onClick={() => openSettings(selectedName)}
              aria-label="Robot settings"
              className="h-7 w-7 rounded-full p-0"
            >
              <Settings className="h-3.5 w-3.5" />
            </Button>
          </span>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {selectedName
            ? `Robot settings for ${selectedName}`
            : "Select a robot first"}
        </TooltipContent>
      </Tooltip>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-2 rounded-full px-2.5 font-medium"
          >
            {isLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : hasRobots && selectedRecord ? (
              <>
                <StatusDot ready={selectedRecord.is_clean} />
                <span className="max-w-[180px] truncate">
                  <span className="text-muted-foreground">Robot: </span>
                  {selectedRecord.name}
                </span>
              </>
            ) : hasRobots ? (
              <span>Select a robot</span>
            ) : (
              <>
                <Plus className="h-3.5 w-3.5" />
                <span>Set up your robot</span>
              </>
            )}
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-72">
          {hasRobots ? (
            <>
              <DropdownMenuLabel className="eyebrow">Robots</DropdownMenuLabel>
              {availableNames.map((name) => {
                const rec = records[name];
                if (!rec) return null;
                const selected = name === selectedName;
                return (
                  <DropdownMenuItem
                    key={name}
                    onSelect={() => selectRobot(name)}
                    className={cn("gap-2", selected && "bg-accent")}
                  >
                    <StatusDot ready={rec.is_clean} />
                    <span className="flex-1 truncate">{name}</span>
                    <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                      {rec.mode === "bimanual" ? "bimanual" : "single"}
                      {" · "}
                      {rec.is_clean ? "ready" : "needs setup"}
                    </span>
                  </DropdownMenuItem>
                );
              })}
              <DropdownMenuSeparator />
            </>
          ) : (
            <DropdownMenuLabel className="text-sm font-normal text-muted-foreground">
              No robots yet. Create one to get started — you'll set up ports,
              calibration, and cameras next.
            </DropdownMenuLabel>
          )}
          <DropdownMenuItem onSelect={() => setCreateOpen(true)} className="gap-2">
            <Plus className="h-4 w-4" />
            Create robot…
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!selectedName}
            onSelect={openRename}
            className="gap-2"
          >
            <Pencil className="h-4 w-4" />
            Rename robot…
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!selectedName}
            onSelect={() => setDeleteOpen(true)}
            className="gap-2 text-destructive focus:text-destructive"
          >
            <Trash2 className="h-4 w-4" />
            Delete robot…
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Tooltip>
        <TooltipTrigger asChild>
          <span>
            <Button
              size="sm"
              variant="secondary"
              className="h-7 gap-1.5 rounded-full px-2.5"
              disabled={!!teleopDisabledReason || teleopStarting}
              onClick={() => selectedRecord && handleTeleop(selectedRecord)}
            >
              {teleopStarting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Gamepad2 className="h-3.5 w-3.5" />
              )}
              Teleop
            </Button>
          </span>
        </TooltipTrigger>
        {teleopDisabledReason && (
          <TooltipContent side="bottom">{teleopDisabledReason}</TooltipContent>
        )}
      </Tooltip>

      <CreateRobotDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        availableNames={availableNames}
        defaultMode="single"
        onCreateNew={handleCreate}
      />

      <TeleopDialog open={teleopOpen} onOpenChange={setTeleopOpen} />

      <RobotConfigDialog
        open={configOpen}
        onOpenChange={handleConfigOpenChange}
        robotName={configRobotName}
      />

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Rename robot</DialogTitle>
            <DialogDescription>
              Calibration assignments, ports, and cameras move with the robot.
            </DialogDescription>
          </DialogHeader>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleRenameConfirm();
            }}
            className="space-y-4"
          >
            <div>
              <Label htmlFor="rename-robot-name">New name</Label>
              <Input
                id="rename-robot-name"
                autoFocus
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                className="mt-1"
              />
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setRenameOpen(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={renaming || !renameValue.trim()}>
                {renaming ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Renaming…
                  </>
                ) : (
                  "Rename"
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete {selectedName ?? "robot"}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This removes the robot's saved configuration (ports, calibration
              assignments, cameras). Calibration files themselves stay in the
              library. This can't be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete robot
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default RobotCorner;
