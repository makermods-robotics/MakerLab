import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import LandingTopBar from "@/components/landing/LandingTopBar";
import Footer from "@/components/Footer";
import { POLICY_TYPE_OPTIONS } from "@/components/training/types";
import {
  fetchPolicyAvailability,
  PolicyAvailability,
} from "@/lib/policyAvailability";
import { useApi } from "@/contexts/ApiContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";

/**
 * Train-a-model page: the policy-type GRID (moved off the Landing panel). A
 * policy tile navigates to /training with the chosen policy in router state (as
 * it did on Landing). A dataset must be selected on the home page first — the
 * tiles gate on it, same as they did inline on Landing.
 *
 * Reached via the Landing "Add model" chooser's "Train a model" entry. The
 * import entry that used to live here was absorbed by that chooser ("Import
 * from disk" COPIES a checkpoint into the local models dir; "Add from Hugging
 * Face" pins a Hub repo). The register-a-pointer import (/jobs/import) remains
 * available in the Jobs section's ImportModelModal.
 */
const CreateModel: React.FC = () => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { selectedDataset } = useSelectedDataset();

  // Which policy types this backend's lerobot pin can actually train. Buttons
  // stay enabled until the (cached) answer arrives — same optimism as Landing.
  const [policyAvailability, setPolicyAvailability] =
    useState<PolicyAvailability | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchPolicyAvailability(baseUrl, fetchWithHeaders)
      .then((a) => {
        if (!cancelled) setPolicyAvailability(a);
      })
      .catch(() => {
        // Backend unreachable — leave buttons enabled; training start surfaces
        // the real error.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  // A policy tile is a direct entry into training: the Training page reads
  // `policyType` from router state and preselects it in the config form.
  const handleTrainingClick = (policyType: string) =>
    navigate("/training", { state: { policyType } });

  return (
    <div className="min-h-screen bg-black text-white pb-16">
      <LandingTopBar />

      <main className="mx-auto max-w-3xl px-4 py-6">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="mb-4 inline-flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" /> Back
        </button>

        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 flex flex-col gap-4">
          <div className="text-center">
            <h1 className="font-semibold text-2xl">Train a model</h1>
            <p className="text-sm text-gray-400 mt-1">
              Train a new policy on your selected dataset.
            </p>
          </div>

          {/* Stable = tested on our hardware (see POLICY_TYPE_OPTIONS).
              Untested types stay selectable, just visually subdued. */}
          <div className="grid grid-cols-2 gap-2">
            {POLICY_TYPE_OPTIONS.filter((p) => p.stable).map((policy) => {
              const unavailable = policyAvailability?.[policy.value] === false;
              return (
                // Tooltip lives on a wrapper span: the disabled Button gets
                // pointer-events-none, which would swallow `title`.
                <span
                  key={policy.value}
                  title={
                    unavailable
                      ? "Not available in this lerobot version"
                      : `Train a ${policy.label} model`
                  }
                >
                  <Button
                    onClick={() => handleTrainingClick(policy.value)}
                    disabled={!selectedDataset || unavailable}
                    size="sm"
                    className="w-full bg-green-500 hover:bg-green-600 text-white px-2"
                  >
                    <span className="truncate">{policy.label}</span>
                  </Button>
                </span>
              );
            })}
          </div>
          <p className="text-xs text-gray-500 -mt-2">
            Untested in MakerLab — use at your own risk
          </p>
          <div className="grid grid-cols-3 gap-2">
            {POLICY_TYPE_OPTIONS.filter((p) => !p.stable).map((policy) => {
              const unavailable = policyAvailability?.[policy.value] === false;
              return (
                <span
                  key={policy.value}
                  title={
                    unavailable
                      ? "Not available in this lerobot version"
                      : `Train a ${policy.label} model — untested in MakerLab, use at your own risk`
                  }
                >
                  <Button
                    onClick={() => handleTrainingClick(policy.value)}
                    disabled={!selectedDataset || unavailable}
                    size="sm"
                    variant="outline"
                    className="w-full border-gray-600 bg-gray-900/40 text-gray-400 hover:bg-gray-700 hover:text-white px-2"
                  >
                    <span className="truncate">{policy.label}</span>
                  </Button>
                </span>
              );
            })}
          </div>
          {!selectedDataset && (
            <p className="text-xs text-gray-500">
              Select a dataset on the home page first.
            </p>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default CreateModel;
