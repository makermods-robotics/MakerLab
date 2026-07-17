import React from "react";
import { disableMockHub, mockHubEnabled } from "@/lib/mockHub";

/**
 * Impossible-to-miss indicator that the dev-only Hub mock (`?mockHub=1`) is
 * serving canned jobs/models/auth instead of real data. Renders nothing in
 * prod builds or when the mock is off. The flag is per-tab (sessionStorage),
 * so no state subscription is needed — it can't change without a navigation
 * that remounts the app anyway.
 */
const MockHubBanner: React.FC = () => {
  if (!mockHubEnabled()) return null;
  return (
    <div className="fixed inset-x-0 bottom-0 z-[100] flex items-center justify-center gap-3 bg-warn px-4 py-1.5 text-xs font-semibold text-white shadow-2">
      <span>
        MOCK HUB DATA — jobs, models, and Hugging Face auth on this page are
        fake (dev fixture).
      </span>
      <button
        type="button"
        onClick={disableMockHub}
        className="rounded border border-current px-2 py-0.5 font-semibold underline-offset-2 hover:underline"
      >
        Turn off
      </button>
    </div>
  );
};

export default MockHubBanner;
