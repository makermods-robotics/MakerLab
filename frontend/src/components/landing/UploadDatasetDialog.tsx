import React, { useState } from "react";
import { Loader2, Upload as UploadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
      >
        <div className="space-y-3">
          <div className="flex items-start gap-2">
            <Checkbox
              id={`hub-upload-private-${repoId}`}
              checked={isPrivate}
              onCheckedChange={(c) => setIsPrivate(c as boolean)}
              className="mt-0.5"
            />
            <Label
              htmlFor={`hub-upload-private-${repoId}`}
              className="cursor-pointer font-normal leading-snug text-gray-300"
            >
              Private dataset
              <span className="mt-0.5 block text-gray-500">
                Recordings include your camera footage.
              </span>
            </Label>
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
