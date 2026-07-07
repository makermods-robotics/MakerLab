import React, { useState } from "react";
import { Globe, Loader2, Lock, Upload as UploadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useToast } from "@/hooks/use-toast";

/**
 * Confirm-and-upload popover for pushing a locally-cached dataset to the Hub.
 * Private-by-default toggle (with the camera-footage note) + optional
 * comma-separated tags. The upload runs in the background (see UploadManager /
 * useDatasetUpload): this dialog only kicks it off and closes — the caller's
 * card/row then shows the polled "Uploading…" state and fires the success /
 * error toast on completion. A refused start (409: already running, or the
 * dataset is busy being written) surfaces here as a destructive toast.
 *
 * Shared by DatasetInfoCard's HubSyncRow and the DatasetPicker rows so both
 * offer the identical dialog. The caller owns the upload hook and passes down
 * `start` (which returns an error message, or null on a clean start).
 */
const UploadDatasetDialog: React.FC<{
  repoId: string;
  /** Kicks off the background upload. Resolves to null on a clean start, or an
   * error message when the start was refused / unreachable. */
  start: (tags: string[], isPrivate: boolean) => Promise<string | null>;
  /** Popover trigger element (a button/icon). */
  children: React.ReactNode;
  align?: "start" | "center" | "end";
}> = ({ repoId, start, children, align = "end" }) => {
  const { toast } = useToast();
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);
  const [tagsInput, setTagsInput] = useState("");
  const [starting, setStarting] = useState(false);

  const handleUpload = async () => {
    setStarting(true);
    try {
      const tags = tagsInput
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);
      const error = await start(tags, isPrivate);
      if (error) {
        toast({
          title: "Upload failed",
          description: error,
          variant: "destructive",
        });
        return;
      }
      // Clean start: close the popover. The caller's row shows the live
      // "Uploading…" state and toasts on completion.
      setPopoverOpen(false);
      toast({
        title: "Upload started",
        description: `${repoId} is uploading to the Hub in the background.`,
      });
    } finally {
      setStarting(false);
    }
  };

  return (
    <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        align={align}
        className="w-72 border-gray-700 bg-gray-900 text-xs text-gray-200"
        // This popover is portaled to the body, but React synthetic events
        // still bubble through the React tree to the cmdk CommandItem that
        // renders our trigger (see DatasetPicker). cmdk's CommandItem selects
        // the row on click, so a click on any control in here (the
        // Public/Private toggle buttons, the Tags input, the Upload button)
        // would otherwise select the dataset and close the picker. Stop the
        // click (and pointer-down) from reaching cmdk. We only stopPropagation
        // — never preventDefault — so focusing the input and the buttons' own
        // handlers keep working. Radix's own outside-click / Escape close is
        // driven by events *outside* this content, so it's unaffected.
        onClick={(e) => e.stopPropagation()}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label
              id={`hub-upload-visibility-${repoId}`}
              className="font-normal text-gray-400"
            >
              Visibility
            </Label>
            <div
              role="radiogroup"
              aria-labelledby={`hub-upload-visibility-${repoId}`}
              className="flex rounded-md border border-gray-700 bg-gray-800 p-0.5"
            >
              <button
                type="button"
                role="radio"
                aria-checked={!isPrivate}
                onClick={() => setIsPrivate(false)}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors ${
                  !isPrivate
                    ? "bg-gray-600 text-white"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                <Globe className="h-3 w-3" />
                Public
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={isPrivate}
                onClick={() => setIsPrivate(true)}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors ${
                  isPrivate
                    ? "bg-gray-600 text-white"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                <Lock className="h-3 w-3" />
                Private
              </button>
            </div>
            <p className="leading-snug text-gray-500">
              {isPrivate
                ? "Only you can see this dataset."
                : "Anyone can see this dataset — recordings include your camera footage."}
            </p>
          </div>
          <div className="space-y-1">
            <Label
              htmlFor={`hub-upload-tags-${repoId}`}
              className="font-normal text-gray-400"
            >
              Tags (optional, comma-separated)
            </Label>
            <Input
              id={`hub-upload-tags-${repoId}`}
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              placeholder="robotics, manipulation"
              className="h-7 border-gray-600 bg-gray-800 text-xs text-white"
            />
          </div>
          <Button
            size="sm"
            onClick={handleUpload}
            disabled={starting}
            className="h-7 w-full gap-1 bg-blue-500 text-xs text-white hover:bg-blue-600"
          >
            {starting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Starting…
              </>
            ) : (
              <>
                <UploadIcon className="h-3 w-3" />
                Upload to Hub
              </>
            )}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default UploadDatasetDialog;
