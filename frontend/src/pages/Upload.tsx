import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { AppShell } from "@/components/shell/AppShell";
import { ArrowRight, CheckCircle, Trash2 } from "lucide-react";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";

interface RecordedDatasetInfo {
  dataset_repo_id: string;
  // Set by the recording flow when the session saved zero episodes and the
  // backend discarded the (empty) dataset directory — nothing is on disk.
  discarded_empty?: boolean;
}

/**
 * Post-recording landing page. Recording flow routes here (see Recording.tsx)
 * once a session ends. Upload no longer happens here — it moved to the dataset
 * info card on the home page, alongside the Hub sync status. This page is now a
 * thin pointer: it confirms the dataset was saved locally and sends the user
 * home with that dataset preselected, so the info card (and its "Upload to Hub"
 * action) is one click away. The /upload route is kept so the recording flow's
 * navigate("/upload", …) keeps working.
 */
const Upload = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { setSelectedDataset } = useSelectedDataset();

  const recorded = location.state?.datasetInfo as
    | RecordedDatasetInfo
    | undefined;
  const discardedEmpty = recorded?.discarded_empty ?? false;
  // A discarded (empty) session left nothing on disk, so there's no repo id to
  // link or preselect.
  const repoId = discardedEmpty ? null : recorded?.dataset_repo_id ?? null;

  const goHome = () => {
    // Preselect the just-recorded dataset so the home page opens straight onto
    // its info card (same mechanism training reads — useSelectedDataset).
    if (repoId) setSelectedDataset(repoId);
    navigate("/");
  };

  if (discardedEmpty) {
    return (
      <AppShell fullBleed>
        <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8">
          <Card className="w-full max-w-md space-y-6 p-8 text-center">
            <div className="flex justify-center">
              <Trash2 className="h-12 w-12 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <h1 className="text-2xl">No episodes were recorded</h1>
              <p className="text-muted-foreground">
                Nothing was saved — the empty dataset was discarded so it doesn't
                take up disk space.
              </p>
            </div>
            <Button onClick={goHome} variant="brand">
              Go to home page
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </Card>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell fullBleed>
      <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8">
        <Card className="w-full max-w-md space-y-6 p-8 text-center">
          <div className="flex justify-center">
            <CheckCircle className="h-12 w-12 text-ok" />
          </div>
          <div className="space-y-2">
            <h1 className="text-2xl">Dataset saved locally</h1>
            {repoId && (
              <p className="break-all font-mono text-sm text-muted-foreground">
                {repoId}
              </p>
            )}
            <p className="text-muted-foreground">
              Review it and upload it to the Hub from the home page.
            </p>
          </div>
          <Button onClick={goHome} variant="brand">
            Go to home page
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </Card>
      </div>
    </AppShell>
  );
};

export default Upload;
