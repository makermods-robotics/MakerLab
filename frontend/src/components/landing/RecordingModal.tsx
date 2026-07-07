import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, CheckCircle, ChevronDown } from "lucide-react";
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { RobotRecord } from "@/hooks/useRobots";
import { validateDatasetName } from "@/lib/datasetName";

interface RecordingModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
  datasetName: string;
  setDatasetName: (value: string) => void;
  singleTask: string;
  setSingleTask: (value: string) => void;
  numEpisodes: number;
  setNumEpisodes: (value: number) => void;
  episodeTimeS: number;
  setEpisodeTimeS: (value: number) => void;
  resetTimeS: number;
  setResetTimeS: (value: number) => void;
  streamingEncoding: boolean;
  setStreamingEncoding: (value: boolean) => void;
  cameras: CameraConfig[];
  setCameras: (cameras: CameraConfig[]) => void;
  onStart: () => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>;
}

const RecordingModal: React.FC<RecordingModalProps> = ({
  open,
  onOpenChange,
  robot,
  datasetName,
  setDatasetName,
  singleTask,
  setSingleTask,
  numEpisodes,
  setNumEpisodes,
  episodeTimeS,
  setEpisodeTimeS,
  resetTimeS,
  setResetTimeS,
  streamingEncoding,
  setStreamingEncoding,
  cameras,
  setCameras,
  onStart,
  releaseStreamsRef,
}) => {
  const { auth } = useHfAuth();

  // null when the name is valid; a message otherwise (incl. empty). Mirrors the
  // backend, so Start can't fire a recording the recorder will reject.
  const nameError = validateDatasetName(datasetName);
  const canStart = !!robot && robot.is_clean && nameError === null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] p-8 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-center items-center mb-4">
            <div className="w-8 h-8 bg-destructive rounded-full flex items-center justify-center">
              <span className="text-destructive-foreground font-bold text-sm">
                REC
              </span>
            </div>
          </div>
          <DialogTitle className="text-center text-xl">
            Configure recording
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-6 py-4">
          <DialogDescription className="text-base leading-relaxed text-center">
            Pick a configured robot and dataset parameters for recording.
          </DialogDescription>

          <div className="grid grid-cols-1 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold border-b border-border pb-2">
                Robot configuration
              </h3>
              {!robot ? (
                <Alert className="bg-warn/10 border-warn/50 text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Select and configure a robot on the Landing page before
                    recording.
                  </AlertDescription>
                </Alert>
              ) : !robot.is_clean ? (
                <Alert className="bg-warn/10 border-warn/50 text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>{robot.name}</strong> is missing a calibration.
                    Configure it before recording.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="flex items-center gap-2 text-sm">
                  <CheckCircle className="w-4 h-4 text-ok" />
                  <span className="text-foreground">
                    Recording with <strong>{robot.name}</strong>
                  </span>
                </div>
              )}
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold border-b border-border pb-2">
                Dataset configuration
              </h3>
              <div className="grid grid-cols-1 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="datasetName">Dataset name *</Label>
                  <Input
                    id="datasetName"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    placeholder="my_dataset"
                    aria-invalid={!!datasetName.trim() && nameError !== null}
                    className="aria-[invalid=true]:border-destructive"
                  />
                  {datasetName.trim() && nameError ? (
                    <p className="text-xs text-destructive">{nameError}</p>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      Letters, numbers, <code>.</code> <code>_</code>{" "}
                      <code>-</code> only; start and end with a letter or number.
                    </p>
                  )}
                  {datasetName &&
                    (auth.status === "authenticated" ? (
                      <p className="text-xs text-muted-foreground">
                        Will be saved as{" "}
                        <span className="text-foreground font-mono">
                          {auth.username}/{datasetName}
                        </span>
                      </p>
                    ) : auth.status === "unauthenticated" ? (
                      <p className="text-xs text-warn">
                        Log in to Hugging Face to set the repository owner.
                      </p>
                    ) : null)}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="singleTask">Task description *</Label>
                  <Input
                    id="singleTask"
                    value={singleTask}
                    onChange={(e) => setSingleTask(e.target.value)}
                    placeholder="e.g., pick up the red block and place it on the blue square"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="numEpisodes">Number of episodes</Label>
                  <NumberInput
                    id="numEpisodes"
                    min="1"
                    max="100"
                    value={numEpisodes}
                    onChange={(v) => {
                      if (v !== undefined) setNumEpisodes(v);
                    }}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="episodeTimeS">
                      Episode duration (seconds)
                    </Label>
                    <NumberInput
                      id="episodeTimeS"
                      min="1"
                      value={episodeTimeS}
                      onChange={(v) => {
                        if (v !== undefined) setEpisodeTimeS(v);
                      }}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="resetTimeS">Reset duration (seconds)</Label>
                    <NumberInput
                      id="resetTimeS"
                      min="1"
                      value={resetTimeS}
                      onChange={(v) => {
                        if (v !== undefined) setResetTimeS(v);
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <CameraConfiguration
                cameras={cameras}
                onCamerasChange={setCameras}
                releaseStreamsRef={releaseStreamsRef}
              />
            </div>

            <Collapsible className="space-y-4 group">
              <CollapsibleTrigger className="flex items-center justify-between w-full text-lg font-semibold border-b border-border pb-2 font-display">
                <span>Advanced parameters</span>
                <ChevronDown className="w-4 h-4 transition-transform group-data-[state=open]:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3">
                <div className="flex items-start gap-3">
                  <Checkbox
                    id="streamingEncoding"
                    checked={streamingEncoding}
                    onCheckedChange={(value) =>
                      setStreamingEncoding(value === true)
                    }
                    className="mt-0.5"
                  />
                  <div className="space-y-1">
                    <Label
                      htmlFor="streamingEncoding"
                      className="cursor-pointer"
                    >
                      Streaming video encoding
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Encodes frames in real time during capture so each
                      episode saves almost instantly. Uncheck to fall back to
                      the slower PNG-then-encode flow.
                    </p>
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={onStart}
              disabled={!canStart}
              variant="notch-brand"
              className="w-full sm:w-auto px-10 py-6 text-lg"
            >
              Start recording
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="w-full sm:w-auto px-10 py-6 text-lg"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default RecordingModal;
