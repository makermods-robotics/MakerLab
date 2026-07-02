import React, { useEffect, useState } from "react";
import { ExternalLink, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

interface HfAuthDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Paste-an-HF-token dialog. Used both to sign in when nobody is authenticated
 * and to "Add account…" from the identity dropdown. The token is validated and
 * stored (named by its displayName, shared with the `hf` CLI) and made active
 * by the backend, then the auth context refetches.
 */
const HfAuthDialog: React.FC<HfAuthDialogProps> = ({ open, onOpenChange }) => {
  const { auth, refetch } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset transient state each time the dialog opens.
  useEffect(() => {
    if (open) {
      setToken("");
      setError(null);
    }
  }, [open]);

  const adding = auth.status === "authenticated";

  const handleSave = async () => {
    const trimmed = token.trim();
    if (!trimmed) return;
    setSubmitting(true);
    setError(null);
    try {
      const r = await fetchWithHeaders(`${baseUrl}/hf-auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: trimmed }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      setToken("");
      await refetch();
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white">
        <DialogHeader>
          <DialogTitle className="text-white">
            {adding ? "Add a Hugging Face account" : "Sign in to Hugging Face"}
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Create a token at{" "}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noreferrer"
              className="underline hover:text-gray-200 inline-flex items-center gap-1"
            >
              huggingface.co/settings/tokens
              <ExternalLink className="w-3 h-3" />
            </a>{" "}
            with <span className="font-mono">Write</span> access (so trained
            policies can upload to your account), then paste it below. The token
            is stored in the machine-global HF token store, shared with the{" "}
            <span className="font-mono">hf</span> CLI.
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSave();
          }}
          className="flex gap-2"
        >
          <Input
            type="password"
            placeholder="hf_..."
            value={token}
            onChange={(e) => setToken(e.target.value)}
            className="bg-slate-900 border-slate-600 text-white placeholder:text-slate-500"
            disabled={submitting}
            autoComplete="off"
            autoFocus
          />
          <Button
            type="submit"
            disabled={submitting || !token.trim()}
            className="bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            {submitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving…
              </>
            ) : adding ? (
              "Add account"
            ) : (
              "Sign in"
            )}
          </Button>
        </form>
        {error && <p className="text-xs text-red-300">{error}</p>}
      </DialogContent>
    </Dialog>
  );
};

export default HfAuthDialog;
