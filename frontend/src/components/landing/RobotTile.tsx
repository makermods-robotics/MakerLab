import React, { useState } from "react";
import { Pencil, Plus, Settings, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Eyebrow } from "@/components/ui/eyebrow";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { RobotRecord, RobotMode } from "@/hooks/useRobots";
import { cn } from "@/lib/utils";
import RobotSelector from "./RobotSelector";
import CreateRobotDialog from "./CreateRobotDialog";

interface RobotTileProps {
  robot: RobotRecord | null;
  selectedName: string | null;
  availableNames: string[];
  modeFilter: RobotMode;
  onFilterChange: (mode: RobotMode) => void;
  isLoading: boolean;
  onSelect: (name: string) => void;
  onCreateNew: (name: string, mode: RobotMode) => Promise<boolean>;
  onConfigure: (name: string) => void;
  onTeleop: (robot: RobotRecord) => void;
  onRename: (oldName: string, newName: string) => Promise<boolean>;
  onDelete: (name: string) => void;
}

const MODE_FILTERS: { value: RobotMode; label: string }[] = [
  { value: "single", label: "Single arm" },
  { value: "bimanual", label: "Bimanual" },
];

const RobotTile: React.FC<RobotTileProps> = ({
  robot,
  selectedName,
  availableNames,
  modeFilter,
  onFilterChange,
  isLoading,
  onSelect,
  onCreateNew,
  onConfigure,
  onTeleop,
  onRename,
  onDelete,
}) => {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);

  const openRename = () => {
    if (!robot) return;
    setRenameValue(robot.name);
    setRenameOpen(true);
  };

  const submitRename = async () => {
    if (!robot) return;
    setRenaming(true);
    const ok = await onRename(robot.name, renameValue);
    setRenaming(false);
    if (ok) setRenameOpen(false);
  };
  const status = robot ? (robot.is_clean ? "Ready" : "Needs configuration") : null;
  const teleopDisabled = !robot || !robot.is_clean;
  // Mirrors CalibrationLibrary's conditional amber warning in its delete
  // dialog: only warn about losing assignments when the robot has some.
  const hasAssignments =
    !!robot &&
    [
      robot.leader_port,
      robot.follower_port,
      robot.leader_config,
      robot.follower_config,
      robot.right_leader_port,
      robot.right_follower_port,
      robot.right_leader_config,
      robot.right_follower_config,
    ].some(Boolean);

  return (
    <Card variant="flat" className="p-3 flex flex-col gap-2 relative">
      <Eyebrow>[ Robot ]</Eyebrow>
      {/* Layout filter. Not a mutator — a record's mode is immutable. Picking
          a side only changes which robots the dropdown lists; the active side
          mirrors the selected robot's layout (selection is the source of
          truth), so it doubles as the layout indicator. */}
      <div
        role="radiogroup"
        aria-label="Filter by arm layout"
        className="grid grid-cols-2 gap-1 rounded-md border border-border bg-secondary p-1"
      >
        {MODE_FILTERS.map((opt) => {
          const active = modeFilter === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onFilterChange(opt.value)}
              className={cn(
                "rounded px-3 py-1.5 text-xs font-medium transition-colors",
                active
                  ? "bg-card text-foreground shadow-1"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {opt.label}
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0">
          <RobotSelector
            selectedName={selectedName}
            availableNames={availableNames}
            defaultMode={modeFilter}
            onSelect={onSelect}
            isLoading={isLoading}
          />
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setCreateOpen(true)}
          disabled={isLoading}
          className="h-8 shrink-0"
        >
          <Plus className="w-3.5 h-3.5 mr-1.5" />
          New robot
        </Button>
        {status && (
          <p
            className={`text-xs truncate shrink-0 ${
              robot!.is_clean ? "text-ok" : "text-warn"
            }`}
          >
            {status}
          </p>
        )}
        {robot && (
          <div className="flex items-center gap-1 shrink-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8"
                  onClick={openRename}
                  aria-label="Rename robot"
                >
                  <Pencil className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Rename robot config</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8"
                  onClick={() => onConfigure(robot.name)}
                  aria-label="Configure"
                >
                  <Settings className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Configure (calibrate)</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-destructive hover:bg-destructive/10 hover:text-destructive"
                  onClick={() => setConfirmDelete(true)}
                  aria-label="Delete robot"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete robot config</TooltipContent>
            </Tooltip>
          </div>
        )}
      </div>

      {robot && (
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="w-full">
              <Button
                onClick={() => onTeleop(robot)}
                disabled={teleopDisabled}
                variant={teleopDisabled ? "secondary" : "brand"}
                className="w-full"
              >
                Teleoperation
              </Button>
            </div>
          </TooltipTrigger>
          {teleopDisabled && (
            <TooltipContent>Configure the robot first.</TooltipContent>
          )}
        </Tooltip>
      )}

      <CreateRobotDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        availableNames={availableNames}
        defaultMode={modeFilter}
        onCreateNew={onCreateNew}
      />

      {robot && (
        <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rename robot config</DialogTitle>
              <DialogDescription>
                Renames the saved robot config only. Calibration files are not
                affected and stay reusable.
              </DialogDescription>
            </DialogHeader>
            <Input
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void submitRename();
                }
              }}
              autoFocus
              placeholder="New name"
            />
            <DialogFooter className="flex gap-2 justify-end">
              <Button variant="outline" onClick={() => setRenameOpen(false)}>
                Cancel
              </Button>
              <Button
                disabled={renaming || !renameValue.trim() || renameValue.trim() === robot.name}
                onClick={submitRename}
              >
                {renaming ? "Renaming…" : "Rename"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {robot && (
        <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete robot "{robot.name}"?</DialogTitle>
              <DialogDescription>
                This permanently deletes the saved robot config — you'd have to
                create and configure it again.
              </DialogDescription>
            </DialogHeader>
            {hasAssignments && (
              <p className="text-sm text-warn">
                This robot has ports and calibrations assigned: those
                assignments will be removed. The calibration files themselves
                are kept in the library and stay reusable.
              </p>
            )}
            <DialogFooter className="flex gap-2 justify-end">
              <Button variant="outline" onClick={() => setConfirmDelete(false)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={async () => {
                  setConfirmDelete(false);
                  await onDelete(robot.name);
                }}
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </Card>
  );
};

export default RobotTile;
