import React from "react";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { AlertTriangle, CheckCircle, ChevronDown } from "lucide-react";
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { RobotRecord, robotSetupGap } from "@/hooks/useRobots";
import { validateDatasetName } from "@/lib/datasetName";

interface RecordingFormProps {
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
  pushToHub: boolean;
  setPushToHub: (value: boolean) => void;
  cameras: CameraConfig[];
  setCameras: (cameras: CameraConfig[]) => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>;
}

/**
 * The recording configuration form — ported from the old landing
 * RecordingModal (its logic is preserved verbatim: name validation +
 * namespace-prefix hint, camera config, streaming-encoding toggle). Lifted
 * out of the dialog into the studio Collect panel and restyled to Layout D
 * tokens. Every session records a NEW dataset — appending to an existing one
 * was removed in favor of merging datasets. The Start button lives in
 * CollectPanel, pinned above the dataset library, not in this form.
 */
const RecordingForm: React.FC<RecordingFormProps> = ({
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
  pushToHub,
  setPushToHub,
  cameras,
  setCameras,
  releaseStreamsRef,
}) => {
  const { auth } = useHfAuth();

  // null when the name is valid; a message otherwise (incl. empty). Mirrors the
  // backend, so Start can't fire a recording the recorder will reject.
  const nameError = validateDatasetName(datasetName);

  return (
    <div className="space-y-6">
      <p className="text-sm leading-relaxed text-muted-foreground">
        Name the dataset and set the capture parameters, then start recording
        on the selected robot.
      </p>

      {/* Robot readiness */}
      <div className="space-y-3">
        <h3 className="eyebrow">Robot</h3>
        {!robot ? (
          <div className="flex gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              Select or create a robot from the corner menu before recording.
            </span>
          </div>
        ) : !robot.is_clean ? (
          <div className="flex gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              <strong>{robot.name}</strong> {robotSetupGap(robot)}. Open Robot
              settings before recording.
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            <span className="text-foreground">
              Recording with <strong>{robot.name}</strong>
            </span>
          </div>
        )}
      </div>

      {/* Dataset parameters */}
      <div className="space-y-4">
        <h3 className="eyebrow">Dataset</h3>

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
              Letters, numbers, <code>.</code> <code>_</code> <code>-</code>{" "}
              only; start and end with a letter or number.
            </p>
          )}
          {datasetName &&
            (auth.status === "authenticated" ? (
              <p className="text-xs text-muted-foreground">
                Will be saved as{" "}
                <span className="font-mono text-foreground">
                  {auth.username}/{datasetName}
                </span>
              </p>
            ) : auth.status === "unauthenticated" ? (
              <p className="text-xs text-amber-700 dark:text-amber-300">
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
            <Label htmlFor="episodeTimeS">Episode duration (s)</Label>
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
            <Label htmlFor="resetTimeS">Reset duration (s)</Label>
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

      {/* Cameras */}
      <CameraConfiguration
        cameras={cameras}
        onCamerasChange={setCameras}
        releaseStreamsRef={releaseStreamsRef}
      />

      {/* Advanced */}
      <Collapsible className="group space-y-3">
        <CollapsibleTrigger className="flex w-full items-center justify-between border-b border-border pb-2 text-sm font-semibold text-foreground">
          <span>Advanced parameters</span>
          <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-3">
          <div className="flex items-start gap-3">
            <Checkbox
              id="streamingEncoding"
              checked={streamingEncoding}
              onCheckedChange={(value) => setStreamingEncoding(value === true)}
              className="mt-0.5"
            />
            <div className="space-y-1">
              <Label
                htmlFor="streamingEncoding"
                className="cursor-pointer font-medium"
              >
                Streaming video encoding
              </Label>
              <p className="text-xs text-muted-foreground">
                Encodes frames in real time during capture so each episode saves
                almost instantly. Uncheck to fall back to the slower
                PNG-then-encode flow.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <Checkbox
              id="pushToHub"
              checked={pushToHub}
              onCheckedChange={(value) => setPushToHub(value === true)}
              className="mt-0.5"
            />
            <div className="space-y-1">
              <Label
                htmlFor="pushToHub"
                className="cursor-pointer font-medium"
              >
                Push to Hugging Face Hub
              </Label>
              <p className="text-xs text-muted-foreground">
                Uploads the dataset to your Hugging Face account in the
                background once the session ends. Uncheck to keep it local —
                you can still upload it later from the dataset library.
              </p>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

export default RecordingForm;
