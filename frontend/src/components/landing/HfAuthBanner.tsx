import React, { useState } from "react";
import { AlertCircle, ExternalLink, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

const HfAuthBanner: React.FC = () => {
  const { auth, refetch } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (auth.status === "authenticated" || auth.status === "loading") {
    return null;
  }

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
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="bg-warn/10 border border-warn/50 rounded-lg p-4 mb-6">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-warn flex-shrink-0 mt-0.5" />
        <div className="flex-1 space-y-3">
          <div>
            <p className="text-sm text-foreground font-medium">
              Hugging Face access required for cloud training
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Create a token at{" "}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noreferrer"
                className="underline hover:text-foreground inline-flex items-center gap-1"
              >
                huggingface.co/settings/tokens
                <ExternalLink className="w-3 h-3" />
              </a>
              {" "}with <span className="font-mono">Write</span> access (so trained
              policies can upload to your account), then paste it below.
            </p>
          </div>
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
              disabled={submitting}
              autoComplete="off"
            />
            <Button type="submit" disabled={submitting || !token.trim()}>
              {submitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving…
                </>
              ) : (
                "Save token"
              )}
            </Button>
          </form>
          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>
      </div>
    </div>
  );
};

export default HfAuthBanner;
