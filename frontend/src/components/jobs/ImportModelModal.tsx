import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, Download, Loader2 } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { importModel, jobDisplayName } from "@/lib/jobsApi";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: () => void;
}

const ImportModelModal: React.FC<Props> = ({ open, onOpenChange, onImported }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [source, setSource] = useState("");
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    const src = source.trim();
    if (!src) return;
    setSubmitting(true);
    setError(null);
    try {
      const record = await importModel(
        baseUrl,
        fetchWithHeaders,
        src,
        name.trim() || undefined,
      );
      if (record.already_imported) {
        // Duplicate source: the backend returned the existing entry (id and
        // display alias preserved) instead of registering a second one.
        toast({
          title: "Already imported",
          description: `"${jobDisplayName(record)}" is already in your models.`,
        });
      }
      setSource("");
      setName("");
      onOpenChange(false);
      onImported();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[520px] p-8">
        <DialogHeader>
          <DialogTitle className="text-center text-xl">
            Import a model
          </DialogTitle>
          <DialogDescription className="text-center">
            Point at a local directory or a Hugging Face repo. It appears as a
            job you can run inference on.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="source">Local path or Hugging Face repo id</Label>
            <Input
              id="source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="/path/to/pretrained_model  or  user/my-policy"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="name">Display name (optional)</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My imported policy"
            />
          </div>

          {error ? (
            <Alert className="bg-destructive/10 border-destructive/50 text-destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <div className="flex gap-3 justify-center pt-2">
            <Button
              onClick={handleSubmit}
              disabled={!source.trim() || submitting}
              variant="brand"
              className="px-8"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              {submitting ? "Importing…" : "Import"}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="px-8"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ImportModelModal;
