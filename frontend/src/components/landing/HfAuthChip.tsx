import React, { useState } from "react";
import { Loader2 } from "lucide-react";
import { useHfAuth } from "@/contexts/HfAuthContext";
import HfAuthDialog from "./HfAuthDialog";

const HfAuthChip: React.FC = () => {
  const { auth } = useHfAuth();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (auth.status === "loading") {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 font-mono text-xs text-muted-foreground">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>Checking HF…</span>
      </div>
    );
  }

  if (auth.status === "authenticated") {
    return (
      <div
        className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 font-mono text-xs text-foreground"
        title="Hugging Face authenticated"
      >
        <span className="h-2 w-2 rounded-full bg-ok" aria-hidden="true" />
        <span>{auth.username}</span>
      </div>
    );
  }

  // unauthenticated
  return (
    <>
      <button
        type="button"
        onClick={() => setDialogOpen(true)}
        className="inline-flex items-center gap-2 rounded-full border border-warn/50 bg-warn/10 px-3 py-1 font-mono text-xs text-warn hover:bg-warn/20 transition-colors"
        aria-label="Hugging Face not configured — show login instructions"
      >
        <span className="h-2 w-2 rounded-full bg-warn" aria-hidden="true" />
        <span>HF not configured</span>
      </button>
      <HfAuthDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
};

export default HfAuthChip;
