import React, { useState } from "react";
import { AlertCircle } from "lucide-react";
import { useHfAuth } from "@/contexts/HfAuthContext";
import HfAuthDialog from "./HfAuthDialog";

const HfAuthBanner: React.FC = () => {
  const { auth } = useHfAuth();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (auth.status === "authenticated" || auth.status === "loading") {
    return null;
  }

  // unauthenticated
  return (
    <>
      <div className="bg-amber-950/40 border border-amber-700/60 rounded-lg p-4 mb-6 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-sm text-amber-100">
            Hugging Face Cloud training requires authentication.{" "}
            <button
              type="button"
              onClick={() => setDialogOpen(true)}
              className="underline hover:text-amber-50 font-semibold"
            >
              Log in to HF CLI
            </button>
          </p>
        </div>
      </div>
      <HfAuthDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </>
  );
};

export default HfAuthBanner;
