import React, { useState } from "react";
import { Loader2 } from "lucide-react";
import { useHfAuth } from "@/contexts/HfAuthContext";
import HfAuthDialog from "./HfAuthDialog";

const HfAuthChip: React.FC = () => {
  const { auth } = useHfAuth();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (auth.status === "loading") {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900/60 px-3 py-1 text-xs text-gray-400">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>Checking HF…</span>
      </div>
    );
  }

  if (auth.status === "authenticated") {
    return (
      <div
        className="inline-flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900/60 px-3 py-1 text-xs text-gray-200"
        title="Hugging Face authenticated"
      >
        <span
          className="h-2 w-2 rounded-full bg-emerald-400"
          aria-hidden="true"
        />
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
        className="inline-flex items-center gap-2 rounded-full border border-amber-700/60 bg-amber-950/40 px-3 py-1 text-xs text-amber-100 hover:bg-amber-900/40 transition-colors"
        aria-label="Hugging Face not configured — show login instructions"
      >
        <span
          className="h-2 w-2 rounded-full bg-amber-400"
          aria-hidden="true"
        />
        <span>HF not configured</span>
      </button>
      <HfAuthDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
};

export default HfAuthChip;
