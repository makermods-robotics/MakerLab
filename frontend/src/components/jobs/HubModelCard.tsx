import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
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
import { HubModel, deleteHubModel } from "@/lib/jobsApi";
import { ApiError } from "@/lib/apiClient";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  ExternalLink,
  Lock,
  Play,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";

interface Props {
  model: HubModel;
  /** Called after a successful delete so the parent can drop the card. */
  onDeleted?: () => void;
  /**
   * Run inference / Fine-tune on this untracked Hub repo. The parent lazily
   * auto-imports the repo (registering it as a tracked imported model), then
   * proceeds exactly as it would for an imported-model card — so this card's
   * primary actions match a regular model card without duplicating any flow.
   */
  onAction?: (
    repoId: string,
    action: "inference" | "finetune",
  ) => void | Promise<void>;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Math.max(0, (Date.now() - t) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/**
 * The strongest confirm in the app: deleting a hub model repo destroys the
 * weights on the Hugging Face Hub permanently — not a local record. Require
 * the user to type the full repo id before the red Delete button enables.
 */
const DeleteHubModelDialog: React.FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
  repoId: string;
  onDeleted?: () => void;
}> = ({ open, onOpenChange, repoId, onDeleted }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Reset the field whenever the dialog (re)opens.
  useEffect(() => {
    if (open) {
      setValue("");
      setError(null);
    }
  }, [open]);

  const confirmed = value.trim() === repoId;

  const doDelete = async () => {
    if (!confirmed || deleting) return;
    setDeleting(true);
    setError(null);
    try {
      await deleteHubModel(baseUrl, fetchWithHeaders, repoId);
      toast({
        title: "Model repo deleted",
        description: repoId,
      });
      onOpenChange(false);
      onDeleted?.();
    } catch (e) {
      setError(
        e instanceof ApiError && e.detail
          ? e.detail
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white">
        <DialogHeader>
          <DialogTitle className="text-red-400">Delete model repo</DialogTitle>
          <DialogDescription className="text-gray-400">
            This permanently deletes the model repository and its files from the
            Hugging Face Hub. This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <p className="text-sm text-gray-300">
            Type{" "}
            <code className="rounded bg-gray-800 px-1 py-0.5 font-mono text-xs text-red-300">
              {repoId}
            </code>{" "}
            to confirm.
          </p>
          <Input
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && confirmed) {
                e.preventDefault();
                void doDelete();
              }
            }}
            autoFocus
            autoComplete="off"
            spellCheck={false}
            placeholder={repoId}
            className="bg-gray-800 border-gray-700 text-white font-mono"
          />
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <DialogFooter className="flex gap-2 justify-end">
          <Button
            variant="outline"
            className="border-gray-600 bg-transparent text-gray-200 hover:bg-gray-800 hover:text-white"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            className="bg-red-600 hover:bg-red-700 text-white"
            disabled={!confirmed || deleting}
            onClick={doDelete}
          >
            {deleting ? "Deleting…" : "Delete permanently"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

const HubModelCard: React.FC<Props> = ({ model, onDeleted, onAction }) => {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [acting, setActing] = useState<"inference" | "finetune" | null>(null);
  const url = `https://huggingface.co/${model.repo_id}`;
  const shortName = model.repo_id.includes("/")
    ? model.repo_id.split("/").slice(1).join("/")
    : model.repo_id;

  const runAction = async (
    e: React.MouseEvent,
    action: "inference" | "finetune",
  ) => {
    e.stopPropagation();
    if (!onAction || acting) return;
    setActing(action);
    try {
      await onAction(model.repo_id, action);
    } finally {
      setActing(null);
    }
  };

  return (
    <Card
      onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
      className="bg-slate-800/50 border-slate-700 rounded-xl cursor-pointer hover:border-slate-500 transition-colors"
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-sky-400">
            <Upload className="w-3.5 h-3.5" />
            Uploaded
          </div>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-7 w-7 text-slate-400 hover:text-white"
              aria-label="View on Hub"
            >
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-slate-400 hover:text-red-400"
              aria-label="Delete model repo"
              onClick={(e) => {
                e.stopPropagation();
                setConfirmOpen(true);
              }}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
        <div>
          <div
            className="text-white font-semibold truncate flex items-center gap-1.5"
            title={model.repo_id}
          >
            {model.private ? (
              <Lock className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            ) : null}
            <span className="truncate">{shortName}</span>
          </div>
          <div className="text-xs text-slate-400 truncate" title={model.repo_id}>
            {model.repo_id} · updated {relativeTime(model.last_modified)}
          </div>
        </div>
        {/* Same primary actions as an imported model card. Clicking either
            lazily auto-imports the repo (in the parent) and then runs the
            action — so a model trained on another machine is a first-class
            citizen here. */}
        {onAction ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="icon"
              onClick={(e) => runAction(e, "inference")}
              disabled={acting !== null}
              className="h-8 w-8 bg-green-500 hover:bg-green-600 text-white"
              aria-label="Run inference with this model"
              title="Run inference"
            >
              <Play className="w-4 h-4" />
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => runAction(e, "finetune")}
              disabled={acting !== null}
              className="h-8 gap-1 border-violet-500/50 text-violet-700 dark:text-violet-300 hover:bg-violet-500/10"
              aria-label="Fine-tune a new run from this model's weights"
              title="Fine-tune a new run from this model's weights"
            >
              <Sparkles className="w-3.5 h-3.5" /> Fine-tune
            </Button>
          </div>
        ) : null}
      </CardContent>
      {/* Rendered inside the Card but its own click handling stops propagation
          so opening/closing the dialog never triggers the card's open-in-Hub. */}
      <div onClick={(e) => e.stopPropagation()}>
        <DeleteHubModelDialog
          open={confirmOpen}
          onOpenChange={setConfirmOpen}
          repoId={model.repo_id}
          onDeleted={onDeleted}
        />
      </div>
    </Card>
  );
};

export default HubModelCard;
