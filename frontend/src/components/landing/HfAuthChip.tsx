import React, { useState } from "react";
import { Loader2 } from "lucide-react";
import { useHfAuth } from "@/contexts/HfAuthContext";
import HfAuthDialog from "./HfAuthDialog";

const HfAuthChip: React.FC = () => {
  const { auth } = useHfAuth();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (auth.status === "loading") {
    return (
      <div className="inline-flex h-8 items-center gap-2 rounded-full border border-border bg-muted/60 px-3 text-sm font-medium text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span>Checking HF…</span>
      </div>
    );
  }

  if (auth.status === "authenticated") {
    return (
      <div
        className="inline-flex h-8 items-center gap-2 rounded-full border border-border bg-muted/60 px-3 text-sm font-medium"
        title={`Logged in to Hugging Face as ${auth.username}`}
      >
        <span
          className="h-2 w-2 rounded-full bg-ok"
          aria-hidden="true"
        />
        <span className="max-w-[180px] truncate">
          <span className="text-muted-foreground">Hugging Face: </span>
          {auth.username}
        </span>
      </div>
    );
  }

  // unauthenticated
  return (
    <>
      <button
        type="button"
        onClick={() => setDialogOpen(true)}
        className="inline-flex h-8 items-center gap-2 rounded-full border border-warn/50 bg-warn/10 px-3 text-sm font-medium text-warn hover:bg-warn/20 transition-colors"
        aria-label="Not logged in to Hugging Face — show login instructions"
      >
        <span
          className="h-2 w-2 rounded-full bg-warn"
          aria-hidden="true"
        />
        <span>Log in to Hugging Face</span>
      </button>
      <HfAuthDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
};

export default HfAuthChip;
