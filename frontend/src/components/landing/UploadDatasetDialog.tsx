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
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import { uploadDataset } from "@/lib/replayApi";

/**
 * Confirm-and-upload popover for pushing a locally-cached dataset to the Hub.
 * Private-by-default toggle (with the camera-footage note) + optional
 * comma-separated tags -> POST /upload-dataset. The upload endpoint is
 * synchronous and datasets are large, so the button shows a spinner while in
 * flight; there's no client-side timeout (see uploadDataset).
 *
 * Shared by DatasetInfoCard's HubSyncRow and the DatasetPicker rows so both
 * offer the identical dialog. The caller supplies the trigger element and an
 * `onUploaded` callback (fired with the Hub URL) so it can refresh its own hub
 * state — e.g. flip a "local only" row to "on Hub".
 */
const UploadDatasetDialog: React.FC<{
  repoId: string;
  /** Popover trigger element (a button/icon). */
  children: React.ReactNode;
  /** Fired after a successful upload with the resolved Hub dataset URL. */
  onUploaded?: (url: string) => void;
  align?: "start" | "center" | "end";
}> = ({ repoId, children, onUploaded, align = "end" }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [isPrivate, setIsPrivate] = useState(true);
  const [tagsInput, setTagsInput] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    setUploading(true);
    try {
      const tags = tagsInput
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);
      const result = await uploadDataset(
        baseUrl,
        fetchWithHeaders,
        repoId,
        tags,
        isPrivate,
      );
      if (result.success) {
        const url =
          result.dataset_url ?? `https://huggingface.co/datasets/${repoId}`;
        setPopoverOpen(false);
        onUploaded?.(url);
        toast({
          title: "Uploaded to Hub",
          description: (
            <span>
              {repoId} is now on the Hub.{" "}
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline font-medium"
              >
                View dataset
              </a>
            </span>
          ),
        });
      } else {
        const fallback = "Failed to upload dataset to the Hub.";
        toast({
          title: "Upload failed",
          description: result.docs_url ? (
            <span>
              {result.message || fallback}{" "}
              <a
                href={result.docs_url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline font-medium"
              >
                Open setup guide
              </a>
            </span>
          ) : (
            result.message || fallback
          ),
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Upload failed",
        description:
          e instanceof ApiError && e.detail
            ? e.detail
            : "Could not reach the backend to upload.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
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
            disabled={uploading}
            className="h-7 w-full gap-1 bg-blue-500 text-xs text-white hover:bg-blue-600"
          >
            {uploading ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Uploading…
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
