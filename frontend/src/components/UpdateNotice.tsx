import { useState } from "react";
import { Loader2, Copy, Sparkles, ChevronRight } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useUpdateCheck } from "@/hooks/useUpdateCheck";

/**
 * App-level popup that notifies the user when a newer MakerLab is available on
 * GitHub. Offers a copy-able upgrade command, a best-effort "Update now" button
 * (runs the pip upgrade on the backend), and a "don't ask again" opt-out.
 */
const UpdateNotice = () => {
  const { status, open, dismiss } = useUpdateCheck();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [dontAsk, setDontAsk] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [output, setOutput] = useState<string | null>(null);

  if (!status) return null;

  const behind =
    typeof status.commits_behind === "number" && status.commits_behind > 0
      ? `${status.commits_behind} commit${status.commits_behind === 1 ? "" : "s"} behind`
      : "A new version is available";

  const copyCommand = async () => {
    if (!status.update_command) return;
    try {
      await navigator.clipboard.writeText(status.update_command);
      toast({
        title: "copied",
        description: "Update command copied to clipboard.",
      });
    } catch {
      toast({
        title: "copy failed",
        description: "Select and copy the command manually.",
        variant: "destructive",
      });
    }
  };

  const runUpdate = async () => {
    setUpdating(true);
    setOutput(null);
    try {
      const r = await fetchWithHeaders(`${baseUrl}/system/update`, {
        method: "POST",
      });
      const body: { success: boolean; message: string; output: string } =
        await r.json();
      if (body.success) {
        toast({ title: "updated", description: body.message });
        dismiss(false);
      } else {
        setOutput(body.output || body.message);
        toast({
          title: "update failed",
          description: body.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "update failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setUpdating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o && !updating) dismiss(dontAsk);
      }}
    >
      <DialogContent
        className="max-w-lg border-border bg-card text-card-foreground"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-foreground">
            <Sparkles className="h-5 w-5 text-brand" />
            MakerLab update available
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            You're {behind}.
            <br />
            Update to get the latest fixes and features.
            {status.compare_url && (
              <>
                {" "}
                <a
                  href={status.compare_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-info underline underline-offset-4 hover:opacity-80"
                >
                  see what changed
                </a>
                .
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Badge variant="outline">{behind}</Badge>
          <Collapsible>
            <CollapsibleTrigger className="group flex items-center gap-1.5 font-display text-xs font-semibold text-muted-foreground transition-colors hover:text-foreground">
              <ChevronRight className="h-3.5 w-3.5 transition-transform group-data-[state=open]:rotate-90" />
              update manually
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2">
              <Card variant="flat">
                <CardContent className="flex items-start gap-2 p-3">
                  <code className="min-w-0 flex-1 whitespace-pre-wrap break-all font-mono text-xs text-info">
                    {status.update_command}
                  </code>
                  <Button
                    variant="secondary"
                    size="icon"
                    onClick={copyCommand}
                    title="copy command"
                    className="shrink-0"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            </CollapsibleContent>
          </Collapsible>

          {output && (
            <pre className="max-h-40 overflow-auto rounded-md border border-border bg-primary p-3 font-mono text-xs text-primary-foreground whitespace-pre-wrap">
              {output}
            </pre>
          )}

          <div className="flex items-center justify-between gap-3 pt-1">
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <Checkbox
                checked={dontAsk}
                onCheckedChange={(v) => setDontAsk(v === true)}
              />
              don't ask me again
            </label>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                onClick={() => dismiss(dontAsk)}
                disabled={updating}
              >
                later
              </Button>
              {status.can_auto_update && (
                <Button onClick={runUpdate} disabled={updating}>
                  {updating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      updating…
                    </>
                  ) : (
                    "update now"
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default UpdateNotice;
