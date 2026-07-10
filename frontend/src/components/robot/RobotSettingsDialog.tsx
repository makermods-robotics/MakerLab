import { useEffect, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
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
import { Button } from "@/components/ui/button";
import {
  useRobotSettingsState,
  closeRobotSettings,
} from "@/components/robot/robotSettingsStore";
import { refreshRobots, useRobots } from "@/hooks/useRobots";
import RobotSettingsPanel from "@/components/robot/RobotSettingsPanel";

/**
 * The robot settings surface as a modal, mounted once in App.tsx and driven by
 * the module-level store so any surface (Home, sidebar gear, create flow) can
 * open it via openRobotSettings(name). Wraps the same RobotSettingsPanel the
 * /calibration route renders full-page.
 */
const RobotSettingsDialog = () => {
  const { open, robotName } = useRobotSettingsState();
  const { deleteRobot } = useRobots();
  const [confirmOpen, setConfirmOpen] = useState(false);

  // The panel edits /robots out-of-band; refresh the shared store on every
  // close so already-mounted pages immediately see new torque/ports/cameras.
  const wasOpen = useRef(false);
  useEffect(() => {
    if (wasOpen.current && !open) refreshRobots();
    wasOpen.current = open;
  }, [open]);

  const handleDelete = async () => {
    setConfirmOpen(false);
    if (!robotName) return;
    const ok = await deleteRobot(robotName);
    if (ok) closeRobotSettings();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) closeRobotSettings();
      }}
    >
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <DialogTitle>{robotName ?? "Robot settings"}</DialogTitle>
        </DialogHeader>

        <RobotSettingsPanel robotName={robotName} variant="dialog" />

        <DialogFooter className="sm:justify-start">
          {robotName && (
            <Button
              variant="outline"
              className="border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setConfirmOpen(true)}
            >
              <Trash2 className="h-4 w-4" />
              Delete robot
            </Button>
          )}
        </DialogFooter>

        <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete {robotName}?</AlertDialogTitle>
              <AlertDialogDescription>
                Delete {robotName}? Calibration files are kept on disk.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter className="flex gap-2 justify-end">
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleDelete}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Delete robot
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </DialogContent>
    </Dialog>
  );
};

export default RobotSettingsDialog;
