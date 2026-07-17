import React, { useState } from "react";
import { Plus, Check, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RobotMode } from "@/hooks/useRobots";
import { cn } from "@/lib/utils";

interface CreateRobotDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  availableNames: string[];
  /** Layout the dialog preseeds to (mirrors the active filter). The user can
   * still change it in the dialog. */
  defaultMode: RobotMode;
  /** Optional name to seed the input with (e.g. a fresh name typed in the
   * selector's search box). */
  seedName?: string;
  onCreateNew: (name: string, mode: RobotMode) => Promise<boolean>;
}

const MODE_OPTIONS: { value: RobotMode; label: string; description: string }[] = [
  {
    value: "single",
    label: "Single arm",
    description: "One leader + one follower",
  },
  {
    value: "bimanual",
    label: "Bimanual",
    description: "Two leader/follower pairs (4 arms)",
  },
];

/**
 * Name + arm-layout form for creating a new robot. Extracted from RobotSelector
 * so the same validated flow can be opened from either the selector's in-menu
 * row or a visible "New robot" button on the Landing card. useRobots owns
 * validation, API errors, and toasts; this component only manages the dialog.
 */
const CreateRobotDialog: React.FC<CreateRobotDialogProps> = ({
  open,
  onOpenChange,
  availableNames,
  defaultMode,
  seedName,
  onCreateNew,
}) => {
  const [newName, setNewName] = useState("");
  const [newMode, setNewMode] = useState<RobotMode>(defaultMode);
  const [creating, setCreating] = useState(false);

  const nameExists = (name: string) =>
    availableNames.some((n) => n.toLowerCase() === name.toLowerCase());

  // Seed the form each time the dialog opens: carry a fresh typed name (if
  // any) and preseed the layout to the active filter.
  React.useEffect(() => {
    if (open) {
      const seed = (seedName ?? "").trim();
      setNewName(seed !== "" && !nameExists(seed) ? seed : "");
      setNewMode(defaultMode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const trimmedNewName = newName.trim();
  const newNameExists = trimmedNewName !== "" && nameExists(trimmedNewName);
  const canConfirm = trimmedNewName !== "" && !newNameExists && !creating;

  const handleCreateConfirm = async () => {
    if (!canConfirm) return;
    setCreating(true);
    try {
      // useRobots handles validation, API errors, and toasts; on success it
      // also selects the new robot. We only manage the dialog here.
      const ok = await onCreateNew(trimmedNewName, newMode);
      if (ok) {
        onOpenChange(false);
        setNewName("");
        setNewMode(defaultMode);
      }
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        onOpenChange(o);
        if (!o) {
          setNewName("");
          setNewMode(defaultMode);
        }
      }}
    >
      <DialogContent className="bg-popover border-border sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create a new robot</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Choose a name and arm layout. The layout is fixed once created — a
            bimanual rig is a separate robot.
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleCreateConfirm();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="new-robot-name" className="text-foreground">
              Name
            </Label>
            <Input
              id="new-robot-name"
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="my_robot"
              aria-invalid={newNameExists}
              className="mt-1 aria-[invalid=true]:border-destructive"
            />
            {newNameExists && (
              <p className="mt-1 text-xs text-destructive">
                A robot with this name already exists.
              </p>
            )}
          </div>
          <div>
            <Label className="text-foreground">Arm layout</Label>
            <div
              role="radiogroup"
              aria-label="Arm layout"
              className="mt-1 grid grid-cols-2 gap-2"
            >
              {MODE_OPTIONS.map((opt) => {
                const selected = newMode === opt.value;
                return (
                  <button
                    key={opt.value}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => setNewMode(opt.value)}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left transition-colors",
                      selected
                        ? "border-primary bg-accent"
                        : "border-border bg-card hover:bg-accent"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">
                        {opt.label}
                      </span>
                      {selected && <Check className="h-4 w-4 text-primary" />}
                    </div>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {opt.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={!canConfirm}>
              {creating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Creating…
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-2" /> Create
                </>
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateRobotDialog;
